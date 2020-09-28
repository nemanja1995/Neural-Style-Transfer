"""
Main script for generating neural style transfer.
"""

from src.NeuralStyleModel import NeuralStyleModel

import argparse


def str2bool(v):
    """
    Used for parsing boolean arguments
    """
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def main(args):

    path_content = args.path_content
    path_style = args.path_style
    path_output = args.path_output
    title = args.title
    save_epoch = args.save_epoch

    neural_style = NeuralStyleModel(output_dir=path_output)
    neural_style.set_content_image(path_context_image=path_content)
    neural_style.set_style_image(path_style_image=path_style)

    neural_style.process_neural_style(num_steps=100, num_epochs=8, title=title, save_epoch=save_epoch)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Script for starting neural style generator process.'
                                                 ' It generates NST picture.')
    parser.add_argument('--path_content', type=str, help='Path to content image', default="data/contents/pewdiepie2.png")
    parser.add_argument('--path_style', type=str, help='Path to style image', default="data/styles/van3_style.jpg")
    parser.add_argument('--path_output', type=str, help='Path to output dir', default="data/output")
    parser.add_argument('--title', type=str, help='Output picture title', default='test')
    parser.add_argument('--save_epoch', help='Save each epoch output', default=False, type=str2bool, nargs='?', const=True)
    main(parser.parse_args())

