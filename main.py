import os
import cv2
import time
import torch
import argparse
import numpy as np

from Detection.Utils import ResizePadding
from CameraLoader import CamLoader, CamLoader_Q
from DetectorLoader import TinyYOLOv3_onecls

from PoseEstimateLoader import SPPE_FastPose
from fn import draw_single

from Track.Tracker import Detection, Tracker
from ActionsEstLoader import TSSTG

# source = '../Data/test_video/test7.mp4'
# source = '../Data/falldata/Home/Videos/video (2).avi'  # hard detect
source = 'vd_120.avi'


# source = 2


def preproc(image):
    """preprocess function for CameraLoader.
    """
    image = resize_fn(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def kpt2bbox(kpt, ex=20):
    """Get bbox that hold on all of the keypoints (x,y)
    kpt: array of shape `(N, 2)`,
    ex: (int) expand bounding box,
    """
    return np.array((kpt[:, 0].min() - ex, kpt[:, 1].min() - ex,
                     kpt[:, 0].max() + ex, kpt[:, 1].max() + ex))


if __name__ == '__main__':
    par = argparse.ArgumentParser(description='Human Fall Detection Demo.')
    par.add_argument('-C', '--camera', default=source,  # required=True,  # default=2,
                     help='Source of camera or video file path.')
    par.add_argument('--detection_input_size', type=int, default=384,
                     help='Size of input in detection model in square must be divisible by 32 (int).')
    par.add_argument('--pose_input_size', type=str, default='224x160',
                     help='Size of input in pose model must be divisible by 32 (h, w)')
    par.add_argument('--pose_backbone', type=str, default='resnet50',
                     help='Backbone model for SPPE FastPose model.')
    par.add_argument('--show_detected', default=False, action='store_true',
                     help='Show all bounding box from detection.')
    par.add_argument('--show_skeleton', default=True, action='store_true',
                     help='Show skeleton pose.')
    par.add_argument('--save_out', type=str, default='',
                     help='Save display to video file.')
    par.add_argument('--device', type=str, default='cuda',
                     help='Device to run model on cpu or cuda.')
    args = par.parse_args()

    device = args.device

    # DETECTION MODEL.
    inp_dets = args.detection_input_size

    detect_model = TinyYOLOv3_onecls(inp_dets, device=device)

    # POSE MODEL.
    inp_pose = args.pose_input_size.split('x')
    inp_pose = (int(inp_pose[0]), int(inp_pose[1]))
    pose_model = SPPE_FastPose(args.pose_backbone, inp_pose[0], inp_pose[1], device=device)

    # Tracker.
    max_age = 30
    tracker = Tracker(max_age=max_age, n_init=3)

    # Actions Estimate.
    action_model = TSSTG()

    resize_fn = ResizePadding(inp_dets, inp_dets)

    cam_source = args.camera
    if type(cam_source) is str and os.path.isfile(cam_source):
        # Use loader thread with Q for video file.
        cam = CamLoader_Q(cam_source, queue_size=1000, preprocess=preproc).start()
    else:
        # Use normal thread loader for webcam.
        cam = CamLoader(int(cam_source) if cam_source.isdigit() else cam_source,
                        preprocess=preproc).start()

    # frame_size = cam.frame_size
    # scf = torch.min(inp_size / torch.FloatTensor([frame_size]), 1)[0]

    outvid = False
    if args.save_out != '':
        outvid = True
        codec = cv2.VideoWriter_fourcc(*'MJPG')
        writer = cv2.VideoWriter(args.save_out, codec, 30, (inp_dets * 2, inp_dets * 2))

    fps_time = 0
    f = 0

    lastdetected = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    ii = 0

    while cam.grabbed():
        f += 1
        frame = cam.getitem()
        image = frame.copy()

        # Detect humans bbox in the frame with detector model.
        detected = detect_model.detect(frame, need_resize=False, expand_bb=10)
        # Predict each tracks bbox of current frame from previous frames information with Kalman filter.
        tracker.predict()
        # Merge two source of predicted bbox together.
        for track in tracker.tracks:
            det = torch.tensor([track.to_tlbr().tolist() + [0.5, 1.0, 0.0]], dtype=torch.float32)
            detected = torch.cat([detected, det], dim=0) if detected is not None else det

        detections = []  # List of Detections object for tracking.
        first_filter = []
        if detected is not None:
            if lastdetected is not None:
                # detected = non_max_suppression(detected[None, :], 0.45, 0.2)[0]
                # Predict skeleton pose of each bboxs.
                ccc = min(detected.__len__(), lastdetected.__len__())
                detected_mat = np.array(detected)
                lastdetected_mat = np.array(lastdetected)

                first_filter = list(range(ccc))
                for aa in range(ccc):

                    x_label = (((int(detected_mat[aa, 0]) + int(detected_mat[aa, 2])) / 2) - (
                            (int(lastdetected_mat[aa, 0]) + int(lastdetected_mat[aa, 2])) / 2))
                    y_label = (((int(detected_mat[aa, 1]) + int(detected_mat[aa, 3])) / 2) - (
                            (int(lastdetected_mat[aa, 1]) + int(lastdetected_mat[aa, 3])) / 2))
                    print(x_label)
                    if abs(x_label) > 5 or abs(y_label) > 5:
                        first_filter.remove(aa)
                print(first_filter)

                if not first_filter == []:
                    poses = pose_model.predict(frame, detected[first_filter, 0:4], detected[first_filter, 4])
                    # for ps in poses:
                    #     print("p========:", (ps['keypoints'].numpy()))
                    #     print("p========:", (ps['keypoints'].numpy().shape))
                    #     print(int(ps['keypoints'].numpy()[12][0]))

                    # Create Detections object.
                    detections = [Detection(kpt2bbox(ps['keypoints'].numpy()),
                                            np.concatenate((ps['keypoints'].numpy(),
                                                            ps['kp_score'].numpy()), axis=1),
                                            ps['kp_score'].mean().numpy()) for ps in poses]

                    # VISUALIZE.
                    if args.show_detected:
                        for bb in detected[first_filter:, 0:5]:
                            frame = cv2.rectangle(frame, (bb[0], bb[1]), (bb[2], bb[3]), (0, 0, 255), 1)
                    gray = cv2.imread('image1.jfif', 0)

                    gray = cv2.resize(gray, (384, 216), interpolation=cv2.INTER_CUBIC)
                    sp = gray.shape
                    print('-----', sp)
                    # cv2.imshow('gray', gray)
                    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    for ps in poses:
                        print('x1', int(ps['keypoints'].numpy()[12][0]))
                        print('y1', int(ps['keypoints'].numpy()[12][1]))
                        print('x2', int(ps['keypoints'].numpy()[11][0]))
                        print('y2', int(ps['keypoints'].numpy()[11][1]))
                        cv2.circle(frame, (int(ps['keypoints'].numpy()[11][0]), int(ps['keypoints'].numpy()[11][1])), 20, (0, 0, 255))
                        cv2.circle(gray, (int(ps['keypoints'].numpy()[11][0]), (int(ps['keypoints'].numpy()[11][1]) - 84)),
                                   20, (0, 0, 255))
                        # gray = cv2.resize(gray, (0, 0), fx=2., fy=2.)
                        cv2.imshow('1', gray)

                        right_foot = gray[
                            (int(ps['keypoints'].numpy()[12][1]) - 84), int(ps['keypoints'].numpy()[12][0])]
                        left_foot = gray[
                            (int(ps['keypoints'].numpy()[11][1]) - 84), int(ps['keypoints'].numpy()[11][0])]
                        print('grayx', right_foot)
                        print('grayy', left_foot)
                        if right_foot > 200 or left_foot > 200:
                            # Update tracks by matching each track information of current and previous frame or
                            # create a new track if no matched.
                            tracker.update(detections)

            # Predict Actions of each track.
            for i, track in enumerate(tracker.tracks):
                if not track.is_confirmed():
                    continue

                track_id = track.track_id
                bbox = track.to_tlbr().astype(int)
                # print(bbox)
                center = track.get_center().astype(int)

                action = 'pending..'
                clr = (0, 255, 0)
                # Use 30 frames time-steps to prediction.
                if len(track.keypoints_list) == 30:
                    pts = np.array(track.keypoints_list, dtype=np.float32)
                    out = action_model.predict(pts, frame.shape[:2])
                    action_name = action_model.class_names[out[0].argmax()]
                    action = '{}: {:.2f}%'.format(action_name, out[0].max() * 100)
                    if action_name == 'Fall Down':
                        clr = (255, 0, 0)
                    elif action_name == 'Lying Down':
                        clr = (255, 200, 0)

                # VISUALIZE.
                if track.time_since_update == 0:
                    if args.show_skeleton:
                        frame = draw_single(frame, track.keypoints_list[-1])
                    frame = cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 1)
                    frame = cv2.putText(frame, str(track_id), (center[0], center[1]),
                                        cv2.FONT_HERSHEY_COMPLEX,
                                        0.4, (255, 0, 0), 2)
                    frame = cv2.putText(frame, action, (bbox[0] + 5, bbox[1] + 15),
                                        cv2.FONT_HERSHEY_COMPLEX,
                                        0.4, clr, 1)

        # Show Frame.
        # frame = cv2.resize(frame, (0, 0), fx=2., fy=2.)
        frame = cv2.putText(frame, '%d, FPS: %f' % (f, 1.0 / (time.time() - fps_time)),
                            (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        frame = frame[:, :, ::-1]
        fps_time = time.time()

        if outvid:
            writer.write(frame)

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        ii = ii + 1
        if ii > 10:
            lastdetected = detected
            ii = 0

    # Clear resource.
    cam.stop()
    if outvid:
        writer.release()
    cv2.destroyAllWindows()
