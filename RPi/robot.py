#!/usr/bin/env/python3
# File name   : robot.py
# Description : Robot interfaces.
import time
import json
import serial
import threading

SERIAL_PORT = "/dev/ttyS0"
SERIAL_BAUDRATE = 115200
ser = None
serial_lock = threading.Lock()


def _open_serial():
	global ser
	if ser is not None and ser.is_open:
		return True
	try:
		ser = serial.Serial(SERIAL_PORT, SERIAL_BAUDRATE, timeout=1, write_timeout=1)
		return True
	except Exception as e:
		print('serial open failed:', e)
		ser = None
		return False


def _serial_write(data_cmd):
	global ser
	payload = data_cmd.encode()
	with serial_lock:
		for _ in range(2):
			if not _open_serial():
				time.sleep(0.2)
				continue
			try:
				ser.write(payload)
				return True
			except Exception as e:
				print('serial write failed:', e)
				try:
					ser.close()
				except Exception:
					pass
				ser = None
				time.sleep(0.2)
	return False


_open_serial()
dataCMD = json.dumps({'var':"", 'val':0, 'ip':""})
upperGlobalIP = 'UPPER IP'


pitch, roll = 0, 0


def setUpperIP(ipInput):
	global upperGlobalIP
	upperGlobalIP = ipInput

def forward(speed=100):
	dataCMD = json.dumps({'var':"move", 'val':1})
	_serial_write(dataCMD)
	print('robot-forward')

def backward(speed=100):
	dataCMD = json.dumps({'var':"move", 'val':5})
	_serial_write(dataCMD)
	print('robot-backward')

def left(speed=100):
	dataCMD = json.dumps({'var':"move", 'val':2})
	_serial_write(dataCMD)
	print('robot-left')

def right(speed=100):
	dataCMD = json.dumps({'var':"move", 'val':4})
	_serial_write(dataCMD)
	print('robot-right')

def stopLR():
	dataCMD = json.dumps({'var':"move", 'val':6})
	_serial_write(dataCMD)
	print('robot-stop')

def stopFB():
	dataCMD = json.dumps({'var':"move", 'val':3})
	_serial_write(dataCMD)
	print('robot-stop')



def lookUp():
	dataCMD = json.dumps({'var':"ges", 'val':1})
	_serial_write(dataCMD)
	print('robot-lookUp')

def lookDown():
	dataCMD = json.dumps({'var':"ges", 'val':2})
	_serial_write(dataCMD)
	print('robot-lookDown')

def lookStopUD():
	dataCMD = json.dumps({'var':"ges", 'val':3})
	_serial_write(dataCMD)
	print('robot-lookStopUD')

def lookLeft():
	dataCMD = json.dumps({'var':"ges", 'val':4})
	_serial_write(dataCMD)
	print('robot-lookLeft')

def lookRight():
	dataCMD = json.dumps({'var':"ges", 'val':5})
	_serial_write(dataCMD)
	print('robot-lookRight')

def lookStopLR():
	dataCMD = json.dumps({'var':"ges", 'val':6})
	_serial_write(dataCMD)
	print('robot-lookStopLR')



def steadyMode():
	dataCMD = json.dumps({'var':"funcMode", 'val':1})
	_serial_write(dataCMD)
	print('robot-steady')

def jump():
	dataCMD = json.dumps({'var':"funcMode", 'val':4})
	_serial_write(dataCMD)
	print('robot-jump')

def handShake():
	dataCMD = json.dumps({'var':"funcMode", 'val':3})
	_serial_write(dataCMD)
	print('robot-handshake')



def lightCtrl(colorName, cmdInput):
	colorNum = 0
	if colorName == 'off':
		colorNum = 0
	elif colorName == 'blue':
		colorNum = 1
	elif colorName == 'red':
		colorNum = 2
	elif colorName == 'green':
		colorNum = 3
	elif colorName == 'yellow':
		colorNum = 4
	elif colorName == 'cyan':
		colorNum = 5
	elif colorName == 'magenta':
		colorNum = 6
	elif colorName == 'cyber':
		colorNum = 7
	dataCMD = json.dumps({'var':"light", 'val':colorNum})
	_serial_write(dataCMD)


def buzzerCtrl(buzzerCtrl, cmdInput):
	dataCMD = json.dumps({'var':"buzzer", 'val':buzzerCtrl})
	_serial_write(dataCMD)



if __name__ == '__main__':
    # robotCtrl.moveStart(100, 'forward', 'no', 0)
    # time.sleep(3)
    # robotCtrl.moveStop()
    while 1:
        time.sleep(1)
        pass