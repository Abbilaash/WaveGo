#!/usr/bin/env/python
# File name   : server.py
# Production  : Upper Ctrl for Robots
# Author	  : WaveShare

import time
import threading
import os
import socket
import info

#websocket
import asyncio
import websockets

import json
import app


ipaddr_check = "192.168.137.55"


def ap_thread():
	os.system("sudo create_ap wlan0 eth0 WAVE_BOT 12345678")


def wifi_check():
	global ipaddr_check
	time.sleep(5)
	try:
		s =socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
		s.connect(("1.1.1.1",80))
		ipaddr_check=s.getsockname()[0]
		s.close()
		print(ipaddr_check)
	except:
		ap_threading=threading.Thread(target=ap_thread)   
		ap_threading.setDaemon(True)                     
		ap_threading.start()


async def check_permit(websocket):
	while True:
		recv_str = await websocket.recv()
		cred_dict = recv_str.split(":")
		if cred_dict[0] == "admin" and cred_dict[1] == "123456":
			response_str = "Connected!"
			await websocket.send(response_str)
			return True
		else:
			response_str = "sorry, the username or password is wrong, please submit again"
			await websocket.send(response_str)


async def recv_msg(websocket):
	while True: 
		response = {
			'status' : 'ok',
			'title' : '',
			'data' : None
		}

		data = ''
		data = await websocket.recv()
		try:
			data = json.loads(data)
		except Exception as e:
			print('not A JSON')

		if not data:
			continue

		if isinstance(data,str):
			try:
				flask_app.commandInput(data)
			except Exception as e:
				print('command execution failed:', e)
				response['status'] = 'error'
				response['title'] = 'command_error'
				response['data'] = str(e)
				response = json.dumps(response)
				await websocket.send(response)
				continue

			if 'get_info' == data:
				response['title'] = 'get_info'
				response['data'] = [info.get_cpu_tempfunc(), info.get_cpu_use(), info.get_ram_info()]

			if 'findColor' == data:
				flask_app.modeselect('findColor')
				print('set mode as findColor')

			elif 'scan' == data:
				print('scanning')
				# ds = app.camera_opencv.ultra.checkdist()
				# print(ds)
				radar_send = [[3,60],[10,70],[10,80],[10,90],[10,100],[10,110],[3,120]]
				# radar_send = []
				# for i in range(1,150):
				# 	radar_send.append[ds]
				response['title'] = 'scanResult'
				response['data'] = radar_send
				time.sleep(0.3)
				pass

			elif 'motionGet' == data:
				flask_app.modeselect('watchDog')
				print('set mode as watchDog')

			elif 'stopCV' == data:
				flask_app.modeselect('none')

			#CVFL
			elif 'CVFL' == data:
				flask_app.modeselect('findlineCV')
				print('set mode as findlineCV')

			elif 'CVFLColorSet' in data:
				color = int(data.split()[1])
				flask_app.camera.colorSet(color)

			elif 'CVFLL1' in data:
				pos = int(data.split()[1])
				flask_app.camera.linePosSet_1(pos)

			elif 'CVFLL2' in data:
				pos = int(data.split()[1])
				flask_app.camera.linePosSet_2(pos)

			elif 'CVFLSP' in data:
				err = int(data.split()[1])
				flask_app.camera.errorSet(err)

			elif 'defEC' in data:#Z
				fpv.defaultExpCom()


		elif(isinstance(data,dict)):
			if data['title'] == "findColorSet":
				color = data['data']
				flask_app.colorFindSet(color[0],color[1],color[2])

		if data != "get_info":
			print(data)
			
		response = json.dumps(response)
		await websocket.send(response)


async def main_logic(websocket, path=None):
	await check_permit(websocket)
	await recv_msg(websocket)


async def main():
	global flask_app

	wifi_check()
	flask_app = app.webapp()
	flask_app.startthread()
	flask_app.sendIP(ipaddr_check)

	async with websockets.serve(main_logic, '0.0.0.0', 8888):
		print('waiting for connection...')
		await asyncio.Future()


if __name__ == '__main__':
	try:
		asyncio.run(main())
	except Exception as e:
		print(e)
