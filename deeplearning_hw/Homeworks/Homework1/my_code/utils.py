import win32gui
import win32api
import win32ui
import win32con
import time
import cv2 as cv
import numpy as np
import aircv as ac


def getScreenshot(hwnd):
    l,t,r,b = win32gui.GetWindowRect(hwnd)
    w = r - l
    h = b - t
    wDC = win32gui.GetWindowDC(hwnd)
    dcObj = win32ui.CreateDCFromHandle(wDC)
    cDC = dcObj.CreateCompatibleDC()
    dataBitMap = win32ui.CreateBitmap()
    dataBitMap.CreateCompatibleBitmap(dcObj, w, h)
    cDC.SelectObject(dataBitMap)
    cDC.BitBlt((0,0),(w, h) , dcObj, (0,0), win32con.SRCCOPY)
    ## No longer need to save bitmap image
    #dataBitMap.SaveBitmapFile(cDC, bmp_file_name) 

    # Read into opencv variable
    signedIntsArray = dataBitMap.GetBitmapBits(True)
    img = np.fromstring(signedIntsArray, dtype='uint8')
    img.shape = (h,w,4)
    img = cv.cvtColor(img, cv.COLOR_RGBA2RGB)

    # Free Resources
    dcObj.DeleteDC()
    cDC.DeleteDC()
    win32gui.ReleaseDC(hwnd, wDC)
    win32gui.DeleteObject(dataBitMap.GetHandle())
    
    return img

def pressLeftMouse(hwnd, pos):
    tmp = win32api.MAKELONG(pos[0], pos[1])
    win32gui.SendMessage(hwnd, win32con.WM_LBUTTONDOWN, win32con.MK_LBUTTON, tmp)
    win32gui.SendMessage(hwnd, win32con.WM_LBUTTONUP, 0, tmp)
    print("Left mouse clicked")

def findPicPos(screenshot, filename):
    """
    Output: ret, object_pos
    """
    object_img = cv.imread(filename)
    pos = ac.find_template(screenshot, object_img)
    if pos is None:
        return False, (0,0)
    if pos["confidence"] < 0.85:
        return False, (0,0)
    return True, (int(pos["result"][0]), int(pos["result"][1]))

def findPicLoopReturnPos(hwnd, filename):
    """
    Output: tuple, position of the object
    """
    time.sleep(1)
    while True:
        print("Finding image: ", filename)
        curr_frame = getScreenshot(hwnd)
        ret, object_pos = findPicPos(curr_frame, filename)
        if ret:
            print("Find object at: ", object_pos)
            return object_pos
            break
        time.sleep(0.5)
    return

def findPicLoopAndClick(hwnd, filename):
    time.sleep(1)
    while True:
        print("Finding image: ", filename)
        curr_frame = getScreenshot(hwnd)
        ret, object_pos = findPicPos(curr_frame, filename)
        if ret:
            print("Find object at: ", object_pos)
            pressLeftMouse(hwnd, object_pos)
            break
        time.sleep(0.5)
    return

def findPicLoopWithCounter(hwnd, filename, counter):
    """
    Input: counter, in 0.5 seconds
    Output: bool, whether find the object
    """
    time.sleep(0.3)
    for i in range(counter):
        print("Finding image: ", filename)
        curr_frame = getScreenshot(hwnd)
        ret, object_pos = findPicPos(curr_frame, filename)
        if ret:
            print("Find object at: ", object_pos)
            return True
        time.sleep(0.5)
        
    return False

def moveMouseDown(hwnd, pos1, dist):
    time.sleep(1.25)
    tmp1 = win32api.MAKELONG(pos1[0], pos1[1])
    win32gui.SendMessage(hwnd, win32con.WM_LBUTTONDOWN, win32con.MK_LBUTTON, tmp1)
    for i in range(int(dist/5)):
        pos1 = pos1[0], pos1[1] + 5
        tmp1 = win32api.MAKELONG(pos1[0], pos1[1])
        win32gui.SendMessage(hwnd, win32con.WM_MOUSEMOVE, win32con.MK_LBUTTON, tmp1)
        time.sleep(0.01)
    win32gui.SendMessage(hwnd, win32con.WM_LBUTTONUP, 0, tmp1)


