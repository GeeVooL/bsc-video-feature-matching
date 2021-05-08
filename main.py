#!/usr/bin/env python3

import argparse
import cv2
import numpy as np
from enum import Enum


class ObjectIdentifier:
    """
    Object detector using ORB feature detector and FLANN or BF matchers.
    """

    FLANN_INDEX_LSH = 6

    class Matcher(Enum):
        BRUTE_FORCE = 1
        FLANN = 2

    def __init__(self, matcher: Matcher):
        self.feature_detector = cv2.ORB_create()
        self.image = None
        self.key_points = None
        self.descriptors = None

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
                'checks': 50,
            }
            self.matcher = cv2.FlannBasedMatcher(index_params, search_params)
        else:
            raise AttributeError('Unknown matcher type.')

    def detect(self, image_path):
        image = cv2.imread(image_path)
        if image is None:
            raise AttributeError('Invalid image path.')

        self.image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        self.key_points, self.descriptors = \
            self.feature_detector.detectAndCompute(self.image, None)

    def process(self, stream_path):
        video = cv2.VideoCapture(stream_path)
        if not video.isOpened():
            raise AttributeError('Invalid video stream path.')

        while video.isOpened():
            is_read, frame = video.read()
            if not is_read:
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            vid_key_points, vid_descriptors = \
                self.feature_detector.detectAndCompute(frame, None)

            matches = self.matcher.match(self.descriptors, vid_descriptors)

            # TODO(mdziewulski): change to ratio-based
            matches = sorted(matches, key=lambda x: x.distance)
            good_matches = matches[:30]

            src_points = np.float32(
                [self.key_points[m.queryIdx].pt for m in good_matches]
            ).reshape(-1, 1, 2)

            dst_points = np.float32(
                [vid_key_points[m.trainIdx].pt for m in good_matches]
            ).reshape(-1, 1, 2)

            M, mask = cv2.findHomography(
                src_points, dst_points, cv2.RANSAC, 5.0)

            matches_mask = mask.ravel().tolist()
            h, w = self.image.shape[:2]
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
                self.image,
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


def parse_arguments():
    parser = argparse.ArgumentParser(description='This program tracks object '
                                                 'from the image on the '
                                                 'specified video stream.')
    parser.add_argument('-m', '--matcher',
                        choices=['brute_force', 'flann'],
                        default='brute_force',
                        help='Matcher used object tracking.')
    parser.add_argument('-i', '--image',
                        required=True,
                        help='Path to the image with an object to track.'
                        )
    parser.add_argument('-s', '--stream',
                        required=True,
                        help='Path to the video stream.')
    return parser.parse_args()


def main(args):
    identifier = ObjectIdentifier(
        matcher=ObjectIdentifier.Matcher[args.matcher.upper()])
    identifier.detect(args.image)
    identifier.process(args.stream)


if __name__ == '__main__':
    main(parse_arguments())
