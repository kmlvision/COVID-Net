import argparse
import logging
import os
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from errors import CouldNotReadImageError, FailedToWriteResultsError


def parse_args():
    parser = argparse.ArgumentParser(description='COVID-Net Inference')
    parser.add_argument('--weightspath', default='model', type=str, help='Path to output folder')
    parser.add_argument('--metaname', default='model.meta_eval', type=str, help='Name of ckpt meta file')
    parser.add_argument('--ckptname', default='model-2069', type=str, help='Name of model ckpts')
    parser.add_argument('--imagepath', default='assets/ex-covid.jpeg', type=str,
                        help='Full path to image to be inferred')
    parser.add_argument('--output_csvpath', default='prediction.csv', type=str, help='Path to output csv file')
    parser.add_argument('--output_vispath', default='prediction.png', type=str,
                        help='Path to output visualization file')
    parser.add_argument('--verbose', type=bool, nargs='?', default=False, const=True, help='print verbose output')

    return parser.parse_args()


def predict(path_image, path_metagraph, path_ckpt, path_output_csv=None, path_output_plot=None, tf_cfg=None):
    class_names = ['normal', 'pneumonia', 'COVID-19']

    x = cv2.imread(path_image)
    if x is None:
        raise CouldNotReadImageError('failed to read input image {}'.format(path_image))
    h, w, c = x.shape
    x = x[int(h / 6):, :]
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
    logging.debug('duration prediction: {:0.3f}s'.format(time.time() - t0))
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
            plt.text(i - .20, np.clip(v + 0.01, 0, 0.95),
                     '{:0.3f}'.format(pred[i]),
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
    logging.info('{}: {:.3f}, {}: {:.3f}, {}: {:.3f}'.format(class_names[0], pred[0],
                                                             class_names[1], pred[1],
                                                             class_names[2], pred[2]))

    # add a disclaimer
    disclaimer_header = '**DISCLAIMER**'
    disclaimer_message = 'Do not use this prediction for self-diagnosis.\n' \
                         'You should check with your local authorities for the latest advice on seeking medical' \
                         ' assistance.\n' \
                         'COVID-Net was developed by Vision and Image Processing Research Group, University of' \
                         ' Waterloo and DarwinAI Corp., Canada.' \
                         ' This software is licensed under the GNU Affero General Public License v3.0.\n' \
                         'The source code and changelog is publicly available from' \
                         ' https://github.com/kmlvision/COVID-Net.'
    logging.info(disclaimer_header)
    logging.info(disclaimer_message)

    for p in (path_output_csv, path_output_plot):
        # write disclaimer files
        if p is not None:
            dirname = os.path.abspath(os.path.dirname(p))
            with open(os.path.join(dirname, 'disclaimer.txt'), 'w') as f:
                f.write(disclaimer_header + "\n")
                f.write(disclaimer_message)


if __name__ == "__main__":
    args = parse_args()
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO, format='%(message)s')
        logging.getLogger('tensorflow').setLevel(logging.WARNING)

    predict(
        args.imagepath,
        os.path.join(args.weightspath, args.metaname),
        os.path.join(args.weightspath, args.ckptname),
        args.output_csvpath,
        args.output_vispath
    )
