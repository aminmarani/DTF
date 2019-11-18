import ast
import json
import os
import pickle
import sys

import tensorflow as tf
from sklearn.metrics import accuracy_score

tf.logging.set_verbosity(tf.logging.INFO)

import model_definition


# model_fn with tf.estimator.Estimator function signature
def cnn_model_fn(features, labels, mode, params):
    # ================================
    # common operations for all modes
    # ================================
    is_training = mode == tf.estimator.ModeKeys.TRAIN
    input_size = 100
    n_output_classes = 2

    inputs = tf.reshape(features['x'], shape=[-1, input_size, input_size, 3])

    conv1 = tf.layers.conv2d(inputs, filters=20, kernel_size=5, padding='same', activation=tf.nn.relu)
    pool1 = tf.layers.max_pooling2d(conv1, pool_size=[2, 2], strides=2)

    # Convolutional Layer #2
    # [batch_size, 14, 14, 32] => [batch_size, 7, 7, 64]
    conv2 = tf.layers.conv2d(pool1, filters=30, kernel_size=5, padding='same', activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(conv2, pool_size=[2, 2], strides=2)

    # Flatten tensor into a batch of vectors
    # [batch_size, 7, 7, 64] => [batch_size, 7 * 7 * 64]
    flat3 = tf.layers.flatten(pool2)

    # Dense Layer with dropout
    # [batch_size, 7 * 7 * 64] => [batch_size, 1024]
    dense4 = tf.layers.dense(flat3, units=12, activation=tf.nn.relu)
    dropout4 = tf.layers.dropout(dense4, rate=0.5, training=mode == tf.estimator.ModeKeys.TRAIN)

    # Logits layer
    # [batch_size, 1024] => [batch_size, 10]
    logits = tf.layers.dense(dropout4, units=n_output_classes)

    # ================================
    # prediction & serving mode
    # mode == tf.estimator.ModeKeys.PREDICT == 'infer'
    # ================================
    predicted_classes = tf.argmax(logits, axis=1)
    predictions = {
        'class_id': tf.cast(predicted_classes, dtype=tf.int32),
        'probabilities': tf.nn.softmax(logits, name="softmax_tensor")
    }
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # compute loss
    # labels: integer 0 ~ 9
    # logits: score not probability
    # loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
    onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=2)
    loss = tf.losses.softmax_cross_entropy(
        onehot_labels=onehot_labels, logits=logits)

    # compute evaluation metric
    # accuracy = tf.metrics.accuracy(labels=labels, predictions=predicted_classes, name='acc_op')
    # metrics = {'accuracy': accuracy}            # during evaluation
    # tf.summary.scalar('accuracy', accuracy[1])  # during training

    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions["class_id"])}

    # ================================
    # evaluation mode
    # ================================
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

    # ================================
    # training mode
    # ================================
    if mode == tf.estimator.ModeKeys.TRAIN:

        optimizer = tf.train.AdadeltaOptimizer(learning_rate=0.01, rho=0.95, epsilon=0.00000001)
        #optimizer = tf.train.SyncReplicasOptimizer(optimizer, replicas_to_aggregate=2)
        #grads_and_vars = optimizer.compute_gradients(loss)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())


        #optimizer.apply_gradients()


        #train_op = optimizer.apply_gradients(grads_and_vars, global_step=tf.train.get_global_step())


        #hook = optimizer.make_session_run_hook(params['is_master'], num_tokens=0)
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)


def _get_session_config_from_env_var():
    """Returns a tf.ConfigProto instance that has appropriate device_filters
    set."""

    tf_config = json.loads(os.environ.get('TF_CONFIG', '{}'))

    if (tf_config and 'task' in tf_config and 'type' in tf_config['task'] and
                'index' in tf_config['task']):
        # Master should only communicate with itself and ps
        if tf_config['task']['type'] == 'master':
            return tf.ConfigProto(device_filters=['/job:ps', '/job:master'])
        # Worker should only communicate with itself and ps
        elif tf_config['task']['type'] == 'worker':
            return tf.ConfigProto(device_filters=[
                '/job:ps',
                '/job:worker/task:%d' % tf_config['task']['index']
            ])
    return None


def train(server_role,server_number,epochs):

    directory = os.path.dirname(os.getcwd())



    if server_role=="master" or server_role=="ps":
        file_pi = open(directory+'/chunks/chunk_master.obj', 'r')
        train_data = pickle.load(file_pi)
        file2 = open(directory+'/chunks/chunk_master_labels.obj', 'r')
        train_labels = pickle.load(file2)

    else:

        file_pi = open(directory+'/chunks/chunk'+str(server_number)+'.obj', 'r')
        train_data = pickle.load(file_pi)
        file2 = open(directory+'/chunks/chunk_labels'+str(server_number)+'.obj', 'r')
        train_labels = pickle.load(file2)


    # hyper parameters
    batch_size = 100

    file_name="distributed_servers"
    file = open(file_name,"r")
    cluster=file.read()
    cluster=ast.literal_eval(cluster)

    # cluster = {"worker": ["localhost:2220"],
    #            "ps": ["localhost:2221"],
    #            "master": ['localhost:2222']}
    # cluster = tf.train.ClusterSpec(cluster)

    os.environ['TF_CONFIG'] = json.dumps(
        {'cluster': cluster,
         'task': {'type': server_role, 'index': server_number}})

    directory = os.path.dirname(os.getcwd())

    model_dir = directory+'/master'
    # create run config for estimator
    run_config = tf.estimator.RunConfig(model_dir=model_dir,save_checkpoints_steps=2000, keep_checkpoint_max=None,
                                        session_config=_get_session_config_from_env_var())
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=1000)



    # create the Estimator
    mnist_classifier = tf.estimator.Estimator(
        model_fn=model_definition.cnn_model_fn2,
        model_dir=model_dir,
        config=run_config,
        params={
            'is_master': server_role=="master",

        }

    )



    #optimizer = tf.train.AdadeltaOptimizer(learning_rate=0.01, rho=0.95, epsilon=0.00000001)
    #optimizer = tf.train.SyncReplicasOptimizer(optimizer, replicas_to_aggregate=2)
    #grads_and_vars = optimizer.compute_gradients(loss)
    # train_op = optimizer.minimize(
    #     loss=loss,
    #     global_step=tf.train.get_global_step())


    # optimizer.apply_gradients()


    #train_op = optimizer.apply_gradients(grads_and_vars)


    #sync_replicas_hook = optimizer.make_session_run_hook(server_role == "master",num_tokens=0)

    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_data},
        y=train_labels,
        batch_size=5,
        num_epochs=epochs,
        shuffle=True)



    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_data},
        y=train_labels,
        num_epochs=1,
        shuffle=False)
    eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn)
    train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn)
    tf.estimator.train_and_evaluate(mnist_classifier, train_spec, eval_spec)



    return


def test(model_dir):


    mnist_classifier = tf.estimator.Estimator(
        model_fn=model_definition.cnn_model_fn2,
        model_dir=model_dir,


    )

    file_pi = open(directory+'/chunks/chunk_validation.obj', 'r')
    train_data = pickle.load(file_pi)
    file2 = open(directory+'/chunks/chunk_validation_labels.obj', 'r')
    train_labels = pickle.load(file2)

    predict_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_data},
        num_epochs=1,
        shuffle=False)
    predict_results = list(mnist_classifier.predict(input_fn=predict_input_fn))
    predicted_classes = [p['class_id'] for p in predict_results]
    accuracy_of_testing=accuracy_score(train_labels, predicted_classes)

    return accuracy_of_testing

def main(server_role,server_number,epochs):

    train(server_role,server_number,epochs)
    return


if __name__ == '__main__':
    server_role = sys.argv[1]
    server_number = sys.argv[2]
    epochs=5
    main(server_role,int(server_number),epochs)


    if server_role=="master":
        directory = os.path.dirname(os.getcwd())
        model_dir = directory + '/master'

        accuracy_of_testing=test(model_dir)

    print("done")