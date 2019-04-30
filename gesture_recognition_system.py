import cv2
import numpy as np
import mysql.connector
import time
import datetime
key_pressed=False
#---------------------------------------------------------------------------------------------------------
def f1():
    return ["Q","W","E","R","T","Y","U","I","O","P","A","S","D",'F','G','H','J','K','L','Z','X','C','V','B','N','M']
#-------------------------------------------------------------------------------------------------------------------
clicked=[]
list=f1()
image = cv2.imread('F:\VISION\master_opencv\images\digits.png')
print(image.shape)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
garray = np.array(gray)
print(garray.shape)
small = cv2.pyrDown(image)
# print(small.shape)
#cv2.imshow("training and testing", small)
cells = [np.hsplit(row, 100) for row in np.vsplit(gray, 50)]
x = np.array(cells)
train = x[:, :70].reshape(-1, 400).astype(np.float32)  # Size = (3500,400)
test = x[:, 70:100].reshape(-1, 400).astype(np.float32)  # Size = (1500,400)
k = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
train_labels = np.repeat(k, 350)[:, np.newaxis]
test_labels = np.repeat(k, 150)[:, np.newaxis]
knn = cv2.ml.KNearest_create()
knn.train(train, cv2.ml.ROW_SAMPLE, train_labels)
ret, result, neighbors, distance = knn.findNearest(test, k=1)
#------------------------------------------------------------------------------------
def drawkey(frame):
    i = 0
    a = 0
    font = cv2.FONT_HERSHEY_SIMPLEX
    while (i < 10):
        x1 = 0 + i * 90
        x2 = 0 + (i + 1) * 90
        w1 = 25 + i * 90
        cv2.rectangle(frame, (x1, 50), (x2, 110), (0, 255, 0), 3)
        cv2.putText(frame, list[i], (w1, 99), font, 2, (0, 0, 255), 2, cv2.LINE_8)
        i = i + 1
    while (i < 19):
        x1 = 45 + a * 90
        x2 = 45 + (a + 1) * 90
        w1 = 70 + a * 90
        cv2.rectangle(frame, (x1, 110), (x2, 170), (0, 255, 0), 3)
        cv2.putText(frame, list[i], (w1, 159), font, 2, (0, 0, 255), 2, cv2.LINE_8)
        i = i + 1
        a = a + 1
    a = 0
    while (i < 27):
        x1 = 90 + a * 90
        x2 = 90 + (a + 1) * 90
        w1 = 115 + a * 90
        if (a == 7):
            cv2.rectangle(frame, (x1, 170), (x2 + 45, 230), (0, 255, 0), 3)
            cv2.putText(frame, 'Backspace', (w1 - 20, 210), font, 0.8, (0, 0, 255), 2, cv2.LINE_8)
        else:
            cv2.rectangle(frame, (x1, 170), (x2, 230), (0, 255, 0), 3)
            cv2.putText(frame, list[i], (w1, 219), font, 2, (0, 0, 255), 2, cv2.LINE_8)
        i = i + 1
        a = a + 1
    cv2.rectangle(frame, (180, 230), (630, 290), (0, 255, 0), 3)
#--------------------------------------------------------------------------------------
def return_letter(x,y):
    global key_pressed
    if (171<=y<=230)&(720<=x<=810):
        clicked.pop()
        key_pressed=True
        return ""
    elif 50<=y<=110:
        key=int((x)/90)
        return list[key]
    elif 111<=y<=170:
        key = int((x-45) / 90)
        return list[key+10]
    elif 171<=y<=230:
        key = int((x-90) / 90)
        return list[key+19]
    elif ((180<=x<=630)&(231<=y<=290)):
        return " "
    else:
        return ""
#--------------------------------------------------------------------------------------
def makeSquare(not_square):
    # This function takes an image and makes the dimenions square
    # It adds black pixels as the padding where needed
    BLACK = [0, 0, 0]
    img_dim = not_square.shape
    height = img_dim[0]
    width = img_dim[1]
    # print("Height = ", height, "Width = ", width)
    if (height == width):
        square = not_square
        return square
    else:
        doublesize = cv2.resize(not_square, (2 * width, 2 * height), interpolation=cv2.INTER_CUBIC)
        height = height * 2
        width = width * 2
        # print("New Height = ", height, "New Width = ", width)
        if (height > width):
            pad = int((height - width) / 2)
            # print("Padding = ", pad)
            doublesize_square = cv2.copyMakeBorder(doublesize, 0, 0, pad,pad, cv2.BORDER_CONSTANT, value=BLACK)
            cv2.imshow("not square", not_square)
            cv2.imshow("double size",doublesize_square)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            pad = int((width - height) / 2)
            # print("Padding = ", pad)
            doublesize_square = cv2.copyMakeBorder(doublesize, pad, pad, 0, 0,cv2.BORDER_CONSTANT, value=BLACK)
            cv2.imshow("double size", doublesize_square)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    #doublesize_square_dim = doublesize_square.shape
    # print("Sq Height = ", doublesize_square_dim[0], "Sq Width = ", doublesize_square_dim[1])
    return doublesize_square
#-------------------------------------------------------------------------------------------------------------
def x_cord_contour(contour):
    # This function take a contour from findContours
    # it then outputs the x centroid coordinates
    print("contour area",cv2.contourArea(contour))
    M = cv2.moments(contour)
    return (int(M['m10'] / M['m00']))
#-------------------------------------------------------------
def resize_to_pixel(image):
    # This function then re-sizes an image to the specificied dimenions
    squared = image
    r = float(20) / squared.shape[1]
    dim = (20, int(squared.shape[0] * r))
    resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    return resized
#----------------------------------------------------------------------------------------------------------
def finger_contour(hsv):
    global key_pressed
    global elapsed1
    print("elapsed1 is",elapsed1)
    global elapsed2
    global sendImage
    global gone
    print("gone is", gone)
    lower_yellow = np.array([16, 100, 100])
    upper_yellow = np.array([36, 229, 255])
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    cv2.imshow("mask for keyboard", mask)
    gaus = cv2.GaussianBlur(mask, (5, 5), 0)
    _, contours, _ = cv2.findContours(gaus, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 30000:
            print(area)
            gone = True
            elapsed2 = time.time()
            if ((elapsed2 - elapsed1) > 3 and elapsed2 > (elapsed2 - elapsed1)):
                sendImage = True
            else:
                elapsed1 = 0
            return contour
    key_pressed = False      # this line will be reached only when counter is less than 30,000
    print("no contour has area greater than 24000")
    if (gone):
        elapsed1 = time.time()
        gone = False
    return ""
#-----------------------------------------------------------------------------------------------------------
def judge_image(frame):
    global knn
    image = frame
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # canny=cv2.Canny(blurred,70,140)
    ret, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV)
    _, contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print("length of contours", len(contours))
    cv2.drawContours(image, contours, -1, (0, 0, 255), 2)
    cv2.imshow("blurred", blurred)
    cv2.imshow("thresh", thresh)
    cv2.imshow("contours", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    contours = sorted(contours, key=x_cord_contour, reverse=False)
    b = 0
    number_read = []
    for c in contours:
        # compute the bounding box for the rectangle
        (x, y, w, h) = cv2.boundingRect(c)
        if w >= 5 and h >= 25:
            # cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
            roi = thresh[y:y + h, x:x + w]
            # ret, roi = cv2.threshold(roi, 127, 255, cv2.THRESH_BINARY_INV)
            squared = makeSquare(roi)
            final = resize_to_pixel(squared)
            final_array = final.reshape((1, 400))
            final_array = final_array.astype(np.float32)
            ret, result, neighbours, dist = knn.findNearest(final_array, k=71)
            print("result is", result)
            print("result[0] is", result[0][0])
            number_read.append(int(result[0][0]))
            name = "squared" + str(b)
            cv2.imshow(name, final)
            b = b + 1
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255,), 2)
            cv2.putText(image, str(int(result[0][0])), (x, y + 155), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
    cv2.imshow('image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print("number is")
    print(number_read)
    return number_read
elapsed1=0
elapsed2=0
gone=False
sendImage=False
conn=mysql.connector.connect(user="root",password="",host="localhost",port="3306",database="project_4")
cur=conn.cursor(buffered=True)
print("enter authority")
authority=input()
print("enter username")
username=input()
print("enter password")
password=input()
def insert_data(algoid,data):
    print("data is",data)
    cur.execute("SELECT MAX(id) from data")
    maxid = 0
    for id in cur:
        maxid = int(id[0])
    cur.execute("SELECT userid from users where u_name=%s", (username,))
    now = datetime.datetime.now()
    date = str(now.year) + "-" + str(now.month) + "-" + str(now.day)
    print(date)
    userid = 0
    for id in cur:
        userid = int(id[0])
    print("maxid is", maxid)
    maxid += 1
    cur.execute("INSERT INTO data(id,userid,data,isadmin,algoid,activity) VALUES (%s,%s,%s,%s,%s,%s)",
                (maxid, userid, data, False, algoid, date))
    conn.commit()
try:
    if authority == "admin":
        cur.execute("SELECT * from admins where a_name=%s and password=%s", (username, password))
        print(cur.fetchall())
        print(cur.rowcount)
        if (cur.rowcount > 0):
            print(""" ENTER YOUR CHOICE
                1) all algos available
                2) all interfaces avialable
                3) all users
                4) logged in users
                5) get recent activity
                6) reset system""")
            choice = int(input())
            if choice == 1:
                algos = []
                cur.execute("SELECT * from algos")
                for algo in cur:
                    algos.append(algo[1])
                for algo in algos:
                    print("algorithms are", algo)
            elif choice == 2:
                interfaces = []
                cur.execute("SELECT * from interface")
                for interface in cur:
                    interfaces.append(interface[1])
                for interface in interfaces:
                    print("interfaces are", interface)
            elif choice == 4:
                users = []
                cur.execute("SELECT * from users where loggedin=1")
                if cur.rowcount > 0:
                    for user in cur:
                        users.append(user[1])
                    for user in users:
                        print(" loggedin users are", user)
                else:
                    print("no logged in users")
            elif choice == 3:
                users = []
                cur.execute("SELECT * from users")
                for user in cur:
                    users.append(user[1])
                for user in users:
                    print("users are", user)
            elif choice == 5:
                users = []
                cur.execute("""SELECT  u_name,data,al_name from data,algos,users where 
                data.algoid=algos.algoid AND users.userid=data.userid 
                AND (datediff(CURDATE(),STR_TO_DATE(activity, '%Y-%m-%d'))<=30)""")
                print("user name    data    algorithm")
                for user in cur:
                    print(user[0], "    ", user[1], "    ", user[2])
            elif choice == 6:
                # cur.execute("""DELETE FROM admins""")
                # cur.execute("""DELETE FROM users""")
                # cur.execute("""DELETE FROM data""")
                print("sytem restored to factory settings")

        else:
            print("wrong username and password")
    elif authority == "user":
        cur.execute("SELECT * from users where u_name=%s and password=%s", (username, password))
        if (cur.rowcount > 0):
            print(""" ENTER YOUR CHOICE
                           1) my recent activity
                           2) use gesture driven keyboard
                           3) use handwriting tool""")
            choice = int(input())
            if choice == 1:
                data = []
                userid = 0
                cur.execute("SELECT userid from users where u_name=%s", (username,))
                for id in cur:
                    userid = id[0]
                cur.execute(
                    """SELECT data from data where (datediff(CURDATE(),STR_TO_DATE(activity, '%Y-%m-%d'))<=30) AND userid=%s""",
                    (userid,))
                for act in cur:
                    data.append(act[0])
                for act in data:
                    print("recent activities are", act)
            elif choice == 2:
                cap = cv2.VideoCapture(0)
                while True:
                    ret, frames = cap.read()
                    frame = cv2.resize(frames, (900, 500))
                    frame = cv2.flip(frame, 1)
                    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                    drawkey(frame)
                    # qprint("no of contours", len(contours))
                    # print("area of contour", area)
                    # print("counter drawn")
                    contour = finger_contour(hsv)
                    if type(contour) is not str:
                        M = cv2.moments(contour)
                        cx = int(M['m10'] / M['m00'])
                        cy = int(M['m01'] / M['m00'])
                        cv2.drawContours(frame, contour, -1, (0, 255, 255), 3)
                        cv2.circle(frame, (cx, cy), 3, (255, 0, 0), -1)
                        if (key_pressed == False):
                            print("coordinates of center are", cx, cy)
                            letter = return_letter(cx, cy)
                            if (letter != ""):
                                print(letter + " is pressed", )
                                print("cx is", cx, "cy is", cy)
                                clicked.append(letter)
                                key_pressed = True
                                print("key is pressed", str(key_pressed))
                            else:
                                print("out of keyboard")
                                # print("area of contour", area)
                    cv2.imshow("web cam", frame)
                    if cv2.waitKey(1) == ord('q'):
                        break
                cap.release()
                cv2.destroyAllWindows()
                clicked_str = ''.join(clicked)
                insert_data(400, clicked_str)

            elif choice == 3:
                points = []
                cap = cv2.VideoCapture(0)
                while True:
                    ret, frames = cap.read()
                    frame = cv2.resize(frames, (900, 500))
                    frame = cv2.flip(frame, 1)
                    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                    contour = finger_contour(hsv)
                    if type(contour) is not str:
                        if sendImage:
                            for line in points:
                                cv2.circle(frame, (line[0], line[1]), 12, (0, 0, 0), -1)
                            number = judge_image(frame)
                            numbers = ""
                            for n in number:
                                numbers = numbers + str(n)
                            insert_data(401, numbers)
                            sendImage = False
                            points.clear()
                            elapsed1 = 0
                            gone = False
                        else:
                            M = cv2.moments(contour)
                            cx = int(M['m10'] / M['m00'])
                            cy = int(M['m01'] / M['m00'])
                            points.append([cx, cy])
                            cv2.drawContours(frame, contour, -1, (0, 255, 255), 3)
                            for line in points:
                                cv2.circle(frame, (line[0], line[1]), 12, (0, 0, 0), -1)
                            cv2.circle(frame, (cx, cy), 12, (0, 0, 0), -1)
                    else:
                        for line in points:
                            cv2.circle(frame, (line[0], line[1]), 12, (0, 0, 0), -1)
                    cv2.imshow("web cam", frame)
                    if cv2.waitKey(1) == ord('q'):
                        break
                cap.release()
                cv2.destroyAllWindows()
        else:
            print("wrong username and password")
    cur.close()
    conn.close()
except Exception as e:
    print("exception occured",e)
