import time
import numpy as np
import tensorflow as tf
import os
import argparse
import cv2
import logging
import matplotlib.pyplot as plt
import pandas as pd
from errors import CouldNotReadImageError, FailedToWriteResultsError


def parse_args():
    parser = argparse.ArgumentParser(description='COVID-Net Inference')
    parser.add_argument('--weightspath', default='model', type=str, help='Path to output folder')
    parser.add_argument('--metaname', default='model.meta_eval', type=str, help='Name of ckpt meta file')
    parser.add_argument('--ckptname', default='model-2069', type=str, help='Name of model ckpts')
    parser.add_argument('--imagepath', default='assets/ex-covid.jpeg', type=str, help='Full path to image to be inferenced')
    parser.add_argument('--output_csvpath', default='prediction.csv', type=str, help='Path to output csv file')
    parser.add_argument('--output_vispath', default='prediction.png', type=str, help='Path to output visualization file')
    parser.add_argument('--verbose', type=bool, nargs='?', default=False, const=True, help='print verbose output')

    args = parser.parse_args()
    return args


def predict(path_image, path_metagraph, path_ckpt, path_output_csv=None, path_output_plot=None, tf_cfg=None):
    class_names = ['normal', 'pneumonia', 'COVID-19']

    x = cv2.imread(path_image)
    if x is None:
        raise CouldNotReadImageError('failed to read input image {}'.format(path_image))
    x = cv2.resize(x, (224, 224))
    x = x.astype('float32') / 255.0

    sess = tf.Session(config=tf_cfg)

    tf.get_default_graph()
    saver = tf.train.import_meta_graph(path_metagraph)
    saver.restore(sess, path_ckpt)

    graph = tf.get_default_graph()

    image_tensor = graph.get_tensor_by_name("input_1:0")
    pred_tensor = graph.get_tensor_by_name("dense_3/Softmax:0")

    t0 = time.time()
    pred = sess.run(pred_tensor, feed_dict={image_tensor: np.expand_dims(x, axis=0)})
    logging.debug('duration prediction: {:0.3f}s'.format(time.time()-t0))
    pred = pred[0, :]
    assert len(pred) == 3, 'NN must have 3 outputs'

    if path_output_plot is not None:
        plt.figure(figsize=(5, 5))
        plt.bar(class_names, pred)
        plt.ylim((0, 1))
        plt.ylabel('confidence')
        plt.grid(axis='y', alpha=0.1)
        plt.yticks(np.linspace(0, 1, 11))
        for i, v in enumerate(pred):
            plt.text(i-.20, np.clip(v+0.01, 0, 0.95),
                      '{:0.2f}'.format(pred[i]),
                      fontsize=14,
                      color=(0, 0, 0, 0.75))
        try:
            plt.savefig(path_output_plot)
        except Exception as e:
            logging.critical('failed to write csv file results', exc_info=True)
            raise FailedToWriteResultsError('failed to write plot file results') from e

    if path_output_csv is not None:
        data = {
            'prediction': class_names[pred.argmax()],
            'confidence_' + class_names[0]: [pred[0]],
            'confidence_' + class_names[1]: [pred[1]],
            'confidence_' + class_names[2]: [pred[2]],
        }
        df_global = pd.DataFrame(data=data)
        try:
            df_global.to_csv(path_output_csv, index=False, float_format='%.3f')
        except Exception as e:
            logging.critical('failed to write csv file results', exc_info=True)
            raise FailedToWriteResultsError('failed to write CSV file results') from e

    logging.info('Prediction: {}'.format(class_names[pred.argmax()]))
    logging.info('Confidence')
    logging.info('Normal: {:.3f}, Pneumonia: {:.3f}, COVID-19: {:.3f}'.format(pred[0][0], pred[0][1], pred[0][2]))
    logging.info('**DISCLAIMER**')
    logging.info('Do not use this prediction for self-diagnosis.'
                 ' You should check with your local authorities for the latest advice on seeking medical assistance.')


if __name__ == "__main__":
    args = parse_args()
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO, format='%(message)s')
        logging.getLogger('tensorflow').setLevel(logging.WARNING)

    predict(args.imagepath,
            os.path.join(args.weightspath, args.metaname),
            os.path.join(args.weightspath, args.ckptname),
            args.output_csvpath,
            args.output_vispath
            )
