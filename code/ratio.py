import cv2

# MPII에서 각 파트 번호, 선으로 연결될 POSE_PAIRS
BODY_PARTS = { "Head": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
                "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
                "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "Chest": 14,
                "Background": 15 }

POSE_PAIRS = [ ["Head", "Neck"], ["Neck", "RShoulder"], ["RShoulder", "RElbow"],
                ["RElbow", "RWrist"], ["Neck", "LShoulder"], ["LShoulder", "LElbow"],
                ["LElbow", "LWrist"], ["Neck", "Chest"], ["Chest", "RHip"], ["RHip", "RKnee"],
                ["RKnee", "RAnkle"], ["Chest", "LHip"], ["LHip", "LKnee"], ["LKnee", "LAnkle"] ]
    
# 각 파일 path
# mpi
#protoFile = "C:\\openpose-master\\openpose-master\\models\\pose\\mpi\\pose_deploy_linevec.prototxt"
#weightsFile = "C:\\openpose-master\\openpose-master\\models\\pose\\mpi\\pose_iter_160000.caffemodel"

# coco
protoFile = "C:\\openpose-master\\openpose-master\\models\\pose\\coco\\pose_deploy_linevec.prototxt"
weightsFile = "C:\\openpose-master\\openpose-master\\models\\pose\\coco\\pose_iter_440000.caffemodel"

# body25
#protoFile = "C:\\openpose-master\\openpose-master\\models\\pose\\body_25\\pose_deploy.prototxt"
#weightsFile = "C:\\openpose-master\\openpose-master\\models\\pose\\mpi\\pose_iter_584000.caffemodel"


# 위의 path에 있는 network 불러오기
net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)

# 이미지 읽어오기
image = cv2.imread("C:\\openpose-master\\data\\FB2.jpg")

#두 신체부위 사이의 거리를 구하는 함수
def returnLength(point1, point2):
    return ((point1[0]-point2[0])**2 + (point1[1]-point2[1])**2)**0.5

#상하체의 비율을 구하는 함수
def returnRatio():
    bottom = (returnLength(points[12],points[13])+returnLength(points[11],points[12])+returnLength(points[9],points[10])+returnLength(points[8],points[9]))/2
    top = returnLength([(points[8][0]+points[9][0])/2,(points[8][1]+points[9][1])/2],points[1])+returnLength(points[1],points[0])/2
    return top/bottom


# frame.shape = 불러온 이미지에서 height, width, color 받아옴
imageHeight, imageWidth, _ = image.shape
 
# network에 넣기위해 전처리
inpBlob = cv2.dnn.blobFromImage(image, 1.0 / 255, (imageWidth, imageHeight), (0, 0, 0), swapRB=False, crop=False)
 
# network에 넣어주기
net.setInput(inpBlob)

# 결과 받아오기
output = net.forward()

# output.shape[0] = 이미지 ID, [1] = 출력 맵의 높이, [2] = 너비
H = output.shape[2]
W = output.shape[3]
print("이미지 ID : ", len(output[0]), ", H : ", output.shape[2], ", W : ",output.shape[3]) # 이미지 ID

# 키포인트 검출시 이미지에 그려줌
points = []
for i in range(0,15):
    # 해당 신체부위 신뢰도 얻음.
    probMap = output[0, i, :, :]
 
    # global 최대값 찾기
    minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)

    # 원래 이미지에 맞게 점 위치 변경
    x = (imageWidth * point[0]) / W
    y = (imageHeight * point[1]) / H

    # 키포인트 검출한 결과가 0.1보다 크면(검출한곳이 위 BODY_PARTS랑 맞는 부위면) points에 추가, 검출했는데 부위가 없으면 None으로    
    if prob > 0.1 :    
        cv2.circle(image, (int(x), int(y)), 3, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)       # circle(그릴곳, 원의 중심, 반지름, 색)
        cv2.putText(image, "{}".format(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, lineType=cv2.LINE_AA)
        points.append((int(x), int(y)))
    else :
        points.append(None)

cv2.imshow("Output-Keypoints",image)
cv2.waitKey(0)

# 이미지 복사
imageCopy = image

# 각 POSE_PAIRS별로 선 그어줌 (머리 - 목, 목 - 왼쪽어깨, ...)
for pair in POSE_PAIRS:
    partA = pair[0]             # Head
    partA = BODY_PARTS[partA]   # 0
    partB = pair[1]             # Neck
    partB = BODY_PARTS[partB]   # 1
    
    #print(partA," 와 ", partB, " 연결\n")
    if points[partA] and points[partB]:
        cv2.line(imageCopy, points[partA], points[partB], (0, 255, 0), 2)

print("상하체 비율: 상체가 하체보다 "+str(returnRatio())+" 배 만큼 깁니다.")
cv2.imshow("Output-Keypoints",imageCopy)
cv2.waitKey(0)
cv2.destroyAllWindows()
