import os
import cv2
import numpy as np


class GarbageClassifier():
    
    def __init__(self):
        labelsPath = 'cfg/imagenet.shortnames.list'
        weightsPath = 'cfg/darknet19.weights'
        configPath = 'cfg/darknet19.cfg'
        
        self.LABELS = open(labelsPath).read().strip().split("\n")
        self.net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
        self.ln = self.net.getLayerNames()
        self.ln = [self.ln[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]
        
        self.writer = None
        self.video_capture = None

    def load_category(self, dir):
        class_dict = {}
        for file_name in os.listdir(dir):
            class_name = file_name.split(".txt")[0]
            class_dict[class_name] = []
            with open(dir+'/'+file_name, 'r') as f:
                for line in f:
                    obj = line.strip().split()
                    class_dict[class_name].append(obj)
        return class_dict

    def garbage_message(self, class_dict, class_name, class_score):
        if class_name == 'vase' or class_name == 'pot':
            return '{}, {:.2f}: GO GREEN! Consider planting a plant'.format(class_name, class_score)
        #     elif class_name == 'carton':
        #         return '{}, {:.2f}: If too big, throw it in the main bin downstairs'.format(class_name, class_score)
        else:

            for c, v in class_dict.items():
                for obj in v:

                    if class_name == obj[0].lower().replace('_', ' '):
                        print(c)
                        if c == 'ignore':
                            print(class_name, obj[0].lower().replace('_', ' '), c)
                            message = 'no garbage detected'
                        elif c == 'degradable':
                            message = '{}, {:.2f}: DEGRADABLE WASTE'.format(class_name, class_score)
                        elif c == 'non-degradable':
                            message = '{}, {:.2f}: NON-DEGRADABLE WASTE'.format(class_name, class_score)
                        elif c == 'hazardous':
                            message = '{}, {:.2f}: HAZARDOUS WASTE'.format(class_name, class_score)

                        return message

        return '{}, {:.2f}: Category Unknown!'.format(class_name, class_score)

    def predict(self, frame):
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        self.net.setInput(blob)
        layerOutputs = self.net.forward(self.ln)
        for output in layerOutputs:
            for detection in output:
                scores = [x[0][0] for x in detection]
                idx = np.argmax(scores)
                obj_class = self.LABELS[idx].lower().replace('_', ' ')
                obj_score = scores[idx]
        return obj_class, obj_score


    def video(self):

        class_dict = self.load_category('garbage/')

        self.video_capture = cv2.VideoCapture(0)
        fps = int(self.video_capture.get(cv2.CAP_PROP_FPS))

        (W, H) = (None, None)
 
        while (self.video_capture.isOpened()):    
            ret, frame = self.video_capture.read()

            if ret:

                if W is None or H is None:
                    (H, W) = frame.shape[:2]

                obj_class, obj_score = self.predict(frame)

                garbage_mes = self.garbage_message(class_dict, obj_class, obj_score)

                cv2.putText(frame, garbage_mes, (70,100), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 0, 0), 3)

                cv2.imshow('Garbage Disposal Category', frame)

            if self.writer is None:
                fourcc = cv2.VideoWriter_fourcc(*"MJPG")
                self.writer = cv2.VideoWriter('test.avi', fourcc, int(fps/5), (frame.shape[1], frame.shape[0]), True)
            self.writer.write(frame)


            if ret == False:
                break


        #     Hit 'q' on the keyboard to quit!
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


    def classify(self):

        self.video()

        self.writer.release()    
        self.video_capture.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    garbage = GarbageClassifier()
    garbage.classify()
