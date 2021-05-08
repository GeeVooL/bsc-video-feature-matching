#!/usr/bin/env python3

import argparse
import cv2
import numpy as np
from enum import Enum
from pathlib import Path


class VideoFeatureMatcher:
    """
    Feature matcher using ORB feature detector and FLANN or BF matchers to
    identify an object on the video.
    """

    BEST_KEYPOINTS_LIMIT = 150
    FLANN_INDEX_LSH = 6

    class Matcher(Enum):
        BRUTE_FORCE = 1
        FLANN = 2

    def __init__(self, matcher: Matcher):
        self.feature_detector = cv2.ORB_create()
        self.images = []
        self.key_points = []
        self.descriptors = []

        if matcher == self.Matcher.BRUTE_FORCE:
            self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        elif matcher == self.Matcher.FLANN:
            index_params = {
                'algorithm': self.FLANN_INDEX_LSH,
                'table_number': 6,
                'key_size': 12,
                'multi_probe_level': 1,
            }
            search_params = {
                'checks': 100,
            }
            self.matcher = cv2.FlannBasedMatcher(index_params, search_params)
        else:
            raise ValueError('Unknown matcher type.')

    def detect(self, image_path, verbose=False):
        """
        Detect key points on images from a specified directory.

        :param image_path: A path to directory with JPG files
        :param verbose:  If true, display each image with its key points
                         in a separate window.
        :return: None
        """

        # Load images from path
        path = Path(image_path)
        filenames = path.glob('*.jpg')
        for f in filenames:
            image = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
            if image is not None:
                self.images.append(image)
        if len(self.images) == 0:
            raise AttributeError('Couldn\'t read any images from the '
                                 f'{path} dir.')

        key_points_list = []
        descriptors_list = []
        for image in self.images:
            # Define a mask – strip 12.5% of image from top and bottom
            mask = np.zeros(image.shape)
            mask[int(mask.shape[0] / 8):int(mask.shape[0] * 7 / 8), :] = 1
            mask = mask.astype(np.uint8)

            # Detect and compute features and add them to a unified list
            key_points, descriptors = \
                self.feature_detector.detectAndCompute(image, mask)
            key_points_list.extend(key_points)
            descriptors_list.extend(descriptors)

            # If true, show images with their key points in separate windows
            if verbose:
                kp_image = cv2.drawKeypoints(image, key_points, None)
                cv2.imshow("Image preview (press any key to quit)", kp_image)
                cv2.waitKey(0)

        cv2.destroyAllWindows()
        self.key_points = np.array(key_points_list)
        self.descriptors = np.array(descriptors_list)

    def process(self, stream_path):
        """
        Display a video stream and identify a previously detected object.

        :param stream_path: A path to a video file to process.
        :return: None
        """

        # Check whether detect() was called
        if len(self.key_points) == 0:
            raise RuntimeError('Feature detector is not trained. Run detect() '
                               'first.')

        # Open video stream
        video = cv2.VideoCapture(stream_path)
        if not video.isOpened():
            raise RuntimeError(f'Cannot open {stream_path} video file.')

        # Iterate over video's frames
        while video.isOpened():
            is_read, frame = video.read()
            if not is_read:
                break

            # Calculate key points for a frame
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            vid_key_points, vid_descriptors = \
                self.feature_detector.detectAndCompute(frame, None)

            # Match points from the images to the points on the frame
            matches = self.matcher.match(self.descriptors,
                                         vid_descriptors)
            matches = sorted(matches, key=lambda x: x.distance)
            good_matches = matches[:self.BEST_KEYPOINTS_LIMIT]

            # Find homography between pictures and show the result
            src_points = np.float32(
                [self.key_points[m.queryIdx].pt for m in good_matches]
            ).reshape(-1, 1, 2)

            dst_points = np.float32(
                [vid_key_points[m.trainIdx].pt for m in good_matches]
            ).reshape(-1, 1, 2)

            M, mask = cv2.findHomography(
                src_points, dst_points, cv2.RANSAC, 5.0)

            matches_mask = mask.ravel().tolist()
            h, w = self.images[0].shape[:2]
            pts = np.float32(
                [[0, 0],
                 [0, h - 1],
                 [w - 1, h - 1],
                 [w - 1, 0]]
            ).reshape(-1, 1, 2)

            dst = cv2.perspectiveTransform(pts, M)
            dst += (w, 0)

            draw_params = {
                'matchColor': (0, 255, 0),
                'singlePointColor': None,
                'matchesMask': matches_mask,
                'flags': 2,
            }
            img = cv2.drawMatches(
                self.images[0],  # The first picture is displayed by default
                self.key_points,
                frame,
                vid_key_points,
                good_matches,
                None,
                **draw_params
            )
            img = cv2.polylines(
                img, [np.int32(dst)], True, (255, 0, 0), 3, cv2.LINE_AA)

            cv2.imshow("Detection (press 'q' to quit)", img)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()


def parse_arguments():
    parser = argparse.ArgumentParser(description='This program tracks object '
                                                 'from the image on the '
                                                 'specified video stream.')
    parser.add_argument('-m', '--matcher',
                        choices=['brute_force', 'flann'],
                        default='flann',
                        help='Matcher used object tracking.')
    parser.add_argument('-i', '--images',
                        required=True,
                        help='Path to the images directory with an object to '
                             'track. Script reads all *.jpg files from that '
                             'directory.'
                        )
    parser.add_argument('-s', '--stream',
                        required=True,
                        help='Path to the video stream.')
    parser.add_argument('-v', '--verbose',
                        action='store_true',
                        help='Display images with their feature points.')
    return parser.parse_args()


def main(args):
    matcher = VideoFeatureMatcher(
        matcher=VideoFeatureMatcher.Matcher[args.matcher.upper()])
    matcher.detect(args.images, args.verbose)
    matcher.process(args.stream)


if __name__ == '__main__':
    main(parse_arguments())
