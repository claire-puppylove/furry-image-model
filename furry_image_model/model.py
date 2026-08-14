import os
import typing
import json
import pathlib
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MultiLabelBinarizer
from tensorflow.python.keras.saving import saved_model
from tensorflow.python.keras.protobuf import saved_metadata_pb2
from tensorflow.python.saved_model import loader_impl
from tensorflow.python.platform import gfile

file_path = pathlib.Path(__file__).parent.resolve()

class FurryImageModel:
    def __init__(self, model_path: typing.Union[str, os.PathLike], safe_threshold: float = 1/3, explicit_threshold: float = 2/3) -> None:
        """A wrapper class for using the tagger model.

        Args:
            model_path (str | os.PathLike): The path of the TensorFlow SavedModel model.
        """
        self.contains_saved_model = tf.saved_model.contains_saved_model(model_path)
        if self.contains_saved_model:
            # load tf config for metadata on model
            # new keras 3 does not load saved models, but we can still load the metadata as a dictionary and find the layer name
            metadata_file = pathlib.Path(model_path) / 'keras_metadata.pb'
            metadata = saved_metadata_pb2.SavedMetadata()
            meta_graph_def = loader_impl.parse_saved_model(model_path).meta_graphs[0]
            object_graph_def = meta_graph_def.object_graph_def
            path_to_metadata_pb = metadata_file
            with gfile.GFile(path_to_metadata_pb, 'rb') as f:
                file_content = f.read()
            metadata.ParseFromString(file_content)
            md_root = [node for node in metadata.nodes if node.node_path=='root'][0]
            self.tf_config = json.loads(md_root.metadata)['model_config']['config']
            layer_name_to_path = dict((json.loads(node.metadata)['name'],node.node_path) for node in metadata.nodes)
            batch_input_shape = self.tf_config['layers'][0]['config']['batch_input_shape']['items']
            
            # load models for tagging and in
            # Doesn't work in keras 3 (tensorflow >= 2.16)
            # self.full_model = tf.keras.models.load_model(model_path)
            input_layer = tf.keras.Input(shape=batch_input_shape[1:])
            savedmodel_layer = tf.keras.layers.TFSMLayer(model_path, call_endpoint='serving_default')
            outputs = savedmodel_layer(inputs=input_layer)
            self.full_model = tf.keras.Model(inputs=input_layer, outputs=outputs)
            # load saved model without keras layer info and tie it with dictionary to the layer's original name
            saved_model = tf.saved_model.load(model_path)
            hiddedn_input_layer = tf.keras.Input(shape=batch_input_shape[1:])
            saved_hidden_layer_func = saved_model.__getattribute__(layer_name_to_path['feature_layer'].replace('root.',''))
            saved_hidden_layer = tf.keras.layers.Lambda(saved_hidden_layer_func)
            feature_outputs = saved_hidden_layer(hiddedn_input_layer)
            self.feature_model = tf.keras.Model(inputs=hiddedn_input_layer, outputs=feature_outputs)
        else:
            # load models for tagging and in
            self.full_model = tf.keras.models.load_model(model_path)
            input_layer = self.full_model.layers[0]
            hidden_layer = self.full_model.get_layer('feature_layer')
            self.feature_model = tf.keras.Model(inputs=input_layer.input, outputs=hidden_layer.output)
            # load tf config for metadata on model
            self.tf_config = self.full_model.get_config()
            batch_input_shape = self.tf_config['layers'][0]['config']['batch_input_shape']['items']
        self.img_size = batch_input_shape[1]
        self.channels = batch_input_shape[3]
        self.layer_order = [layer[0][:layer[0].find('_')] if layer[0].find('_') != -1 else layer[0] for layer in self.tf_config['output_layers']]
        
        # load multilabel binarizers for model
        self.mlbs = {}
        self.tags = {}
        
        category_dir = file_path / 'categories'
        category_files = os.listdir(category_dir)
        for category in category_files:
            with open(os.path.join(category_dir, category)) as f:
                tags = f.readlines()
            
            category_name = category[:category.find('_')]
            tags = [[tag.replace('\n', '')] for tag in tags]
            mlb = MultiLabelBinarizer()
            mlb.fit(tags)
            
            self.mlbs.update({category_name:mlb})
            self.tags.update({category_name:mlb.classes_})
        
        self.safe_threshold = safe_threshold
        self.explicit_threshold = explicit_threshold

         
    def _load_image(self, img_path: typing.Union[str, os.PathLike]) -> tf.Tensor:
        """Loads an image from the given path and applies the necessary modifications:
        converting to RGB, resizing to fit the model, and Normalizing the pixel values
        between [0.0, 1.0]
        
        Args:
            img_path (str | os.PathLike): The path for the image

        Returns:
            Tensor: Tensor representation of the Image
        """
        # Read an image from a file
        image_string = tf.io.read_file(str(img_path))
        # Decode it into a dense vector
        # Decode it into a dense vector
        image_decoded = tf.image.decode_image(
            image_string,
            channels=self.channels,
            expand_animations=True,
            )
        # Resize it to fixed shape
        image_resized = tf.image.resize(image_decoded, [self.img_size, self.img_size])
        # Normalize it from [0, 255] to [0.0, 1.0]
        image_normalized = image_resized / 255.0
        ## adding gif or webp usability take 4D tensors and disguise them as batches of 3D tensors
        image_4d = tf.reshape(image_normalized,[-1,self.img_size,self.img_size,self.channels])
        return image_4d

    def _get_rating(self, value: float):
        if value >= self.explicit_threshold:
            return 'explicit'
        elif value >= self.safe_threshold:
            return 'questionable'
        else:
            return 'safe'

    def _normalize_func(self, image, normalized):
        img_mean = normalized[0]
        img_stddev = normalized[1]

        offset = tf.constant(img_mean, shape=[1, 1, 3])
        image -= offset
        scale = tf.constant(img_stddev, shape=[1, 1, 3])
        image /= scale
        return image

        
    def predict_image_tags(self, *img_path: typing.Union[str, os.PathLike],
                           t: float = 0.5,
                           normalized: typing.List[tuple] = [(0.5, 0.5, 0.5), (0.225, 0.225, 0.225)],) -> typing.Any:
        """run full prediction model on image to predict tags

        Args:
            *img_path (str | os.PathLike): The path for each image
            
            t (float, optional): The probablility threshold at which to accept a tag as 
            valid. Defaults to 0.5.

        Returns:
            Any: Returns a list of all the tags predicted for each image.
        """

        if len(img_path) == 0:
            raise ValueError('There must be at least one image given.')

        if t < 0 or t > 1:
            raise ValueError('The threshold must be within [0.0, 1.0]')
        
        loaded_images = [self._load_image(img) for img in img_path]
        # loaded as 4d, flatten to list of 3d, keep information of frames>1 index
        num_frames_per_image = [len(img) for img in loaded_images]
        loaded_frames = []
        frame_to_img_index = {}
        frame_ind = 0
        for image_ind, load_img in enumerate(loaded_images):
            img_frames = tf.unstack(load_img)
            loaded_frames.extend(img_frames)
            for i, frame in enumerate(img_frames):
                frame_to_img_index[frame_ind]={'image_num':image_ind,'frame_num':i+1}
                frame_ind+=1
        if normalized:
            loaded_frames = [self._normalize_func(img, normalized) for img in loaded_frames]
        
        num_images = len(loaded_images)
        total_num_frames = len(loaded_frames)
        frames = tf.stack(loaded_frames)
        x = self.full_model.predict(frames)

        res = [{'num_frames':num_f,'r':[{'frame':i_f+1, 'fr':[]} for i_f in range(num_f)]} for num_f in num_frames_per_image]
        # for layer, result in zip(self.layer_order, x):
        for ind, layer in enumerate(self.layer_order):
            if self.contains_saved_model:
                # for some reason x called from TFSMLayer is in the form dict[layer_name:output].
                result = x[layer]
            else:
                # the nth of x is the layer in order
                result = x[ind]
            assert len(result) == total_num_frames
            
            if layer == 'rating':
                for i_frame, r in enumerate(result):
                    current_f = frame_to_img_index[i_frame]['frame_num']
                    current_img = frame_to_img_index[i_frame]['image_num']
                    out = [{'tag': self._get_rating(r[0]),
                            'value': float(r[0]),
                            'category': layer,
                            }]
                    res[current_img]['r'][current_f-1]['fr'].extend(out)
            else:
                result = np.array(result)
                result_mask = np.where(result > t, 1, 0)
                mlb = self.mlbs.get(layer)
                
                mlb: MultiLabelBinarizer
                tags = mlb.inverse_transform(result_mask)
                values = [result[i, np.where(result_mask[i])][0] for i in range(total_num_frames)]
                for i_frame,(t_list, v_list) in enumerate(zip(tags, values)):
                    current_f = frame_to_img_index[i_frame]['frame_num']
                    current_img = frame_to_img_index[i_frame]['image_num']
                    out = [{'tag': t,
                             'value': float(v),
                             'category': layer,
                            }
                            for t,v in zip(t_list, v_list)
                        ]
                    res[current_img]['r'][current_f-1]['fr'].extend(out)

        return res
        
        
    def image_latent_vector(self, *img_path: typing.Union[str, os.PathLike]) -> np.ndarray:
        """Get the feature vector representation of an image.
        
        NOTE: The output array will be shaped (N, D), where N is the flattened # of frames in all images, and D 
        is the dimension of the feature vector.

        Args:
            *img_path (str | os.PathLike): The path for each image
            
        Returns:
            np.ndarray: A dense vector representation of the images.
        """
        if len(img_path) == 0:
            raise ValueError('There must be at least one image given.')

        loaded_images = [self._load_image(img) for img in img_path]
        # loaded as 4d, flatten to list of 3d
        loaded_frames = []
        for load_img in loaded_images:
            img_frames = tf.unstack(load_img)
            loaded_frames.extend(img_frames)
        
        res = self.feature_model.predict(tf.stack(loaded_frames))
        return np.array(res)