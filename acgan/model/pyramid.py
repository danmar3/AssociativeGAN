import typing
import twodlearn as tdl
import tensorflow as tf


@tdl.core.create_init_docstring
class ImagePyramid(tdl.core.Layer):
    @tdl.core.InputArgument
    def scales(self, value: typing.Union[int, typing.List[int]]):
        '''scaling factor.'''
        return value

    @tdl.core.InputArgument
    def depth(self, value: typing.Union[None, int]):
        '''number of scaling operations performed'''
        tdl.core.assert_initialized(self, 'depth', ['scales'])
        if value is None:
            if isinstance(value, int):
                raise ValueError('depth of ImagePyramid not specified')
            value = len(self.scales)
        return value

    @tdl.core.InputArgument
    def interpolation(self, value: typing.Union[None, str]):
        '''Type of interpolation. Either bilinear or neirest (Default) '''
        if value is None:
            value = 'nearest'
        elif value not in ('nearest', 'bilinear', 'bicubic'):
            raise ValueError('interpolation should be nearest, bilinear or '
                             'bicubic')
        return value

    @tdl.core.Submodel
    def _resize_fn(self, _):
        tdl.core.assert_initialized(self, '_resize_fn', ['interpolation'])
        if self.interpolation == 'nearest':
            value = tf.compat.v1.image.resize_nearest_neighbor
        elif self.interpolation == 'bilinear':
            value = tf.compat.v1.image.resize_bilinear
        elif self.interpolation == 'bicubic':
            value = tf.compat.v1.image.resize_bicubic
        return value

    def compute_output_shape(self, input_shape):
        input_shape = input_shape.as_list()
        scales = ([self.scales]*self.depth if isinstance(self.scales, int)
                  else self.scales)
        output_shapes = list()
        current_size = input_shape[1:3]
        for scale in scales:
            if isinstance(scale, int):
                scale = (scale, scale)
            output_shapes.append(tf.TensorShape(
                (input_shape[0],
                 scale[0]*current_size[0], scale[1]*current_size[1],
                 input_shape[-1])
            ))
            current_size = output_shapes[-1].to_list()[1:3]
        return output_shapes

    def call(self, inputs):
        assert self.input_shape.ndims == 4, \
            'provided input is not 4 dimensional'
        scales = ([self.scales]*self.depth if isinstance(self.scales, int)
                  else self.scales)

        output = list([inputs])
        for scale in scales:
            if isinstance(scale, int):
                scale = (scale, scale)
            input_size = tf.convert_to_tensor(inputs).shape.as_list()[1:3]
            size = (input_size[0]//scale[0], input_size[1]//scale[1])
            output.append(self._resize_fn(inputs, size=size))
            inputs = output[-1]
        return output[::-1]
