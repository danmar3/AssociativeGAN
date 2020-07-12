import twodlearn as tdl
import twodlearn.bayesnet
import tensorflow as tf
import tensorflow.keras.layers as tf_layers
from types import SimpleNamespace
from .msg_gan import (AffineLayer, Conv2DLayer, LEAKY_RATE)
from ..utils import eager_function, replicate_to_list


class ResConv(tdl.core.Layer):
    '''CNN for recovering the encodding of a given image'''
    # CONV_LAYER = Conv2DLayer
    CONV_LAYER = tf_layers.Conv2D

    embedding_size = tdl.core.InputArgument.required(
        'embedding_size', doc='dimensionality of the embedding.')

    @tdl.core.SubmodelInit(lazzy=True)
    def layers(self, units, kernels, strides, dropout=None, padding='same',
               pooling=None, flatten='global_maxpool', res_layers=2):
        tdl.core.assert_initialized(self, 'layers', ['embedding_size'])

        n_layers = len(units)
        kernels = replicate_to_list(kernels, n_layers)
        strides = replicate_to_list(strides, n_layers)
        padding = (padding if isinstance(padding, (list, tuple))
                   else [padding]*n_layers)
        pooling = replicate_to_list(pooling, n_layers)

        model = tdl.stacked.StackedLayers()
        for i in range(len(units)):
            model.add(tf_layers.BatchNormalization())
            # residual
            residual = tdl.stacked.StackedLayers()
            for ri in range(res_layers):
                if ri != 1:
                    residual.add(tf_layers.BatchNormalization())
                residual.add(self.CONV_LAYER(
                    filters=units[i], strides=strides[i],
                    kernel_size=kernels[i], padding=padding[i]))
                residual.add(tf_layers.LeakyReLU(LEAKY_RATE))
                if dropout is not None:
                    residual.add(tf_layers.Dropout(rate=dropout))
            # resnet
            model.add(tdl.resnet.ResConv2D(residual=residual))
            if i < len(pooling) and pooling[i] is not None:
                model.add(tf_layers.MaxPooling2D(pool_size=pooling[i]))

        if flatten == 'flatten':
            model.add(tf_layers.Flatten())
        elif flatten == 'global_maxpool':
            model.add(tf_layers.GlobalMaxPooling2D())
        else:
            raise ValueError(f'flatten option {flatten} not recognized')
        model.add(tf_layers.BatchNormalization())
        model.add(AffineLayer(units=self.embedding_size))
        return model

    @tdl.core.Submodel
    def output_dist(self, _):
        tdl.core.assert_initialized(self, 'output_dist', ['embedding_size'])
        return tdl.bayesnet.NormalModel(
            loc=lambda x: x,
            batch_shape=self.embedding_size)

    def call(self, inputs, expected=True):
        tdl.core.assert_initialized(self, 'call', ['layers'])
        output = self.layers(inputs)
        if expected:
            output = self.output_dist(output)
        return output


class CallWrapper(tdl.core.Layer):
    model = tdl.core.Submodel.required('model', doc='wrapped model')
    call_fn = tdl.core.Submodel.required('call_fn', doc='call function')

    def call(self, inputs, **kargs):
        return self.call_fn(self.model, inputs, **kargs)


class LinearClassifier(tdl.core.Layer):
    n_classes = tdl.core.InputArgument.required(
        'n_classes', doc='number of classes.')

    @tdl.core.SubmodelInit(lazzy=True)
    def encoder(self, embedding_size, layers):
        return CallWrapper(
            model=ResConv(embedding_size=embedding_size, layers=layers),
            call_fn=lambda model, x: model(x, expected=False))

    @tdl.core.SubmodelInit(lazzy=True)
    def classifier(self, **kargs):
        tdl.core.assert_initialized(self, 'classifier', ['n_classes'])
        return tf_layers.Dense(units=self.n_classes)

    def call(self, inputs, softmax=False):
        encoded = self.encoder(inputs)
        logits = self.classifier(encoded)
        if softmax:
            return tf.nn.softmax(logits)
        else:
            return logits


class MlpClassifier(LinearClassifier):
    @tdl.core.SubmodelInit(lazzy=True)
    def classifier(self, units):
        tdl.core.assert_initialized(self, 'classifier', ['n_classes'])
        model = tdl.stacked.StackedLayers()
        for i in range(len(units)):
            # model.add(tf_layers.BatchNormalization())
            model.add(tf_layers.Dense(units=units[i]))
            model.add(tf_layers.LeakyReLU(LEAKY_RATE))
        model.add(tf_layers.Dense(units=self.n_classes))
        return model


class Accuracy(tdl.core.Layer):
    from_logits = tdl.core.InputArgument.required(
        'from_logits', doc='inputs are logits.')
    sparse = tdl.core.InputArgument.optional(
        'sparse', doc='labels are sparse.', default=False)

    def call(self, labels, inputs):
        if inputs.shape.as_list()[-1] == 1:
            if self.from_logits:
                inputs = tf.nn.sigmoid(inputs)
            if not self.sparse:
                labels = tf.round(labels, axis=-1)
            comp = tf.equal(labels, tf.round(inputs, axis=-1))
        else:
            if self.from_logits:
                inputs = tf.nn.softmax(inputs)
            if not self.sparse:
                labels = tf.argmax(labels, axis=-1)
            comp = tf.equal(labels, tf.argmax(inputs, axis=-1))
        return tf.reduce_mean(tf.cast(comp, tf.float32))


class Estimator(tdl.core.TdlModel):
    model = tdl.core.Submodel.required('model', doc='model to be trained.')

    @tdl.core.SubmodelInit
    def loss(self, value, from_logits=False):
        if isinstance(value, str):
            if value == 'crossentropy':
                return tf.keras.losses.CategoricalCrossentropy(
                    from_logits=from_logits)
            elif value == 'sparse_crossentropy':
                return tf.keras.losses.SparseCategoricalCrossentropy(
                    from_logits=from_logits)
            else:
                raise ValueError(f'loss string {value} not recognized.')
        else:
            return value

    @tdl.core.SubmodelInit
    def metrics(self, metrics=None, from_logits=False, **kargs):
        if metrics is None:
            return dict()
        if isinstance(metrics, (list, tuple)):
            # build a dict using the first charactes as the name for the metric
            metrics = {name[:3]: name for name in metrics}
        assert isinstance(metrics, dict), \
            'metrics should be specified using a dictionary'
        metrics_names = {
            'accuracy': lambda: Accuracy(from_logits=from_logits, **kargs)
            }
        return {name: metrics_names[value]()
                if isinstance(value, str) else value
                for name, value in metrics.items()}

    def _evaluate_graph(self, train_x, train_y, valid_x=None, valid_y=None):
        tdl.core.assert_initialized(
            self, '_evaluate_graph', ['model', 'loss', 'metrics'])
        train_pred = self.model(train_x)
        train_loss = self.loss(train_y, train_pred)
        train_metrics = {name: mi(train_y, train_pred)
                         for name, mi in self.metrics.items()}
        train = {'loss': train_loss, 'pred': train_pred,
                 'inputs': train_x, 'labels': train_y}
        if train_metrics:
            train['metrics'] = train_metrics
        else:
            train['metrics'] = None
        if valid_x is None:
            valid = None
        return SimpleNamespace(train=train, valid=valid)

    @tdl.core.SubmodelInit
    def train_ops(self, train_x, train_y, valid_x=None, valid_y=None):
        return self._evaluate_graph(train_x, train_y, valid_x, valid_y)

    def compile(self, inputs_spec, labels_spec):
        self.train_ops.init(
            train_x=tf.compat.v1.placeholder(
                inputs_spec.dtype, shape=inputs_spec.shape),
            train_y=tf.compat.v1.placeholder(
                labels_spec.dtype, shape=labels_spec.shape),
            )
        tdl.core.initialize_variables(self.model)

    @tdl.core.SubmodelInit(lazzy=True)
    def trainable_variables(self, get_trainable=None):
        if get_trainable is None:
            tdl.core.assert_initialized(
                self, 'trainable_variables', ['train_ops'])
            value = tdl.core.get_trainable(self.model)
        elif callable(get_trainable):
            value = get_trainable()
        else:
            raise ValueError('get_trainable must be callable')
        return value

    def _get_optimizer(self, ops, learning_rate, **kargs):
        tdl.core.assert_initialized(
            self, '_get_optimizer', ['trainable_variables'])
        return tdl.optimv2.SimpleOptimizer(
            loss=ops.train['loss'],
            var_list=self.trainable_variables,
            metrics={'ops': ops.train['metrics'], 'buffer_size': 100},
            learning_rate=learning_rate,
            **kargs)

    def get_optimizer(self, train_x, train_y, valid_x=None, valid_y=None,
                      learning_rate=None, **kargs):
        ops = self._evaluate_graph(train_x, train_y, valid_x, valid_y)
        return self._get_optimizer(ops, learning_rate=learning_rate, **kargs)

    @tdl.core.SubmodelInit(lazzy=True)
    def optimizer(self, learning_rate=None, **kargs):
        tdl.core.assert_initialized(
            self, 'optimizer', ['train_ops', 'trainable_variables'])
        return self._get_optimizer(
            self.train_ops, learning_rate=learning_rate, **kargs)

    def fit(self, train_x, train_y, n_steps=100):
        if not tdl.core.is_property_initialized(self, 'train_ops'):
            self.compile(train_x, train_y)
        tdl.core.assert_initialized(
            self, 'fit', ['train_ops', 'trainable_variables'])

        def feed_tensors():
            inputs, labels = self.session.run((train_x, train_y))
            return {
                self.train_ops.train['inputs']: inputs,
                self.train_ops.train['labels']: labels
            }
        tdl.core.assert_initialized(self, 'fit', ['optimizer'])
        self.optimizer.run(n_steps=n_steps, feed_dict=feed_tensors)

    @eager_function
    def _evaluate(self, inputs):
        return self.model(inputs)

    @eager_function
    def _evaluate_supervised(self, inputs, labels):
        ops = self._evaluate_graph(train_x=inputs, train_y=labels)
        return {'loss': ops.train['loss'], 'pred': ops.train['pred'],
                'metrics': ops.train['metrics']}

    def evaluate(self, inputs, labels=None):
        if labels is None:
            return self._evaluate(inputs=inputs)
        else:
            return self._evaluate_supervised(inputs=inputs, labels=labels)


class LinearEstimator(Estimator):
    @tdl.core.SubmodelInit
    def model(self, n_classes, encoder, **kargs):
        return LinearClassifier(n_classes=n_classes, encoder=encoder, **kargs)


class MlpEstimator(Estimator):
    @tdl.core.SubmodelInit
    def model(self, n_classes, units, encoder, **kargs):
        return MlpClassifier(n_classes=n_classes, encoder=encoder,
                             classifier={'units': units}, **kargs)
