import 'dart:convert';
import 'package:http/http.dart' as http;

class BotApiClient {
  final String ipAddress; // e.g. "192.168.12.1:5000"
  
  BotApiClient(this.ipAddress);

  String get baseUrl => "http://$ipAddress";

  // Check connectivity and get status
  Future<Map<String, dynamic>> getStatus() async {
    final response = await http.get(
      Uri.parse("$baseUrl/api/status"),
    ).timeout(const Duration(seconds: 4));
    
    if (response.statusCode == 200) {
      return jsonDecode(response.body);
    } else {
      throw Exception("Server returned HTTP ${response.statusCode}");
    }
  }

  // Camera Tilt Control
  Future<Map<String, dynamic>> sendTilt(String direction, String action) async {
    final response = await http.post(
      Uri.parse("$baseUrl/api/tilt/$direction/$action"),
    );
    return jsonDecode(response.body);
  }

  // Move Control
  Future<Map<String, dynamic>> sendMove(String action, int speed) async {
    final response = await http.post(
      Uri.parse("$baseUrl/api/move/$action"),
      headers: {"Content-Type": "application/json"},
      body: jsonEncode({"speed": speed}),
    );
    return jsonDecode(response.body);
  }

  // Preset Behaviors
  Future<Map<String, dynamic>> sendDefaultAction(String action) async {
    final response = await http.post(
      Uri.parse("$baseUrl/api/default/$action"),
    );
    return jsonDecode(response.body);
  }

  // Face Detection Toggle
  Future<Map<String, dynamic>> toggleFaceDetection(String action) async {
    final response = await http.post(
      Uri.parse("$baseUrl/api/face/detect/$action"),
    );
    return jsonDecode(response.body);
  }

  // Face Follow Control
  Future<Map<String, dynamic>> toggleFaceFollow(String action, {String name = ""}) async {
    final response = await http.post(
      Uri.parse("$baseUrl/api/face/follow/$action"),
      headers: {"Content-Type": "application/json"},
      body: jsonEncode({"name": name}),
    );
    return jsonDecode(response.body);
  }

  // Rapid Face Capture
  Future<Map<String, dynamic>> captureFaces() async {
    final response = await http.post(
      Uri.parse("$baseUrl/api/face/capture"),
    );
    return jsonDecode(response.body);
  }

  // Save Captured Face
  Future<Map<String, dynamic>> saveFace(String name, List<String> images) async {
    final response = await http.post(
      Uri.parse("$baseUrl/api/face/save"),
      headers: {"Content-Type": "application/json"},
      body: jsonEncode({"name": name, "images": images}),
    );
    return jsonDecode(response.body);
  }

  // Ball Search Toggle
  Future<Map<String, dynamic>> toggleBallSearch(String action) async {
    final response = await http.post(
      Uri.parse("$baseUrl/api/search/ball"),
      headers: {"Content-Type": "application/json"},
      body: jsonEncode({"action": action}),
    );
    return jsonDecode(response.body);
  }

  // Color Tracking
  Future<Map<String, dynamic>> followColor(String color) async {
    final response = await http.post(
      Uri.parse("$baseUrl/api/color/follow/$color"),
    );
    return jsonDecode(response.body);
  }

  // Handwriting/Digit Recognition
  Future<Map<String, dynamic>> detectHandwriting({String? base64Image, bool explain = false}) async {
    final body = <String, dynamic>{};
    if (base64Image != null) {
      body["image"] = base64Image;
    }
    body["explain"] = explain;

    final response = await http.post(
      Uri.parse("$baseUrl/api/detect_digit"),
      headers: {"Content-Type": "application/json"},
      body: jsonEncode(body),
    );
    return jsonDecode(response.body);
  }

  // Chatbot Command text submit
  Future<Map<String, dynamic>> sendChatbotCommand(String command) async {
    final response = await http.post(
      Uri.parse("$baseUrl/api/chatbot/command"),
      headers: {"Content-Type": "application/json"},
      body: jsonEncode({"command": command}),
    ).timeout(const Duration(seconds: 45));
    return jsonDecode(response.body);
  }

  // Chatbot Command audio upload (WAV format)
  Future<Map<String, dynamic>> sendChatbotAudio(String filePath, {String mode = 'chat'}) async {
    final uri = Uri.parse("$baseUrl/api/chatbot/audio?mode=$mode");
    final request = http.MultipartRequest("POST", uri);
    
    request.files.add(
      await http.MultipartFile.fromPath(
        'audio',
        filePath,
      ),
    );
    
    final streamedResponse = await request.send().timeout(const Duration(seconds: 45));
    final response = await http.Response.fromStream(streamedResponse);
    
    if (response.statusCode == 200) {
      return jsonDecode(response.body);
    } else {
      throw Exception("Audio upload failed (${response.statusCode}): ${response.body}");
    }
  }

  // Bluetooth Scan Nearby Devices
  Future<Map<String, dynamic>> scanBluetoothDevices() async {
    final response = await http.get(
      Uri.parse("$baseUrl/api/bluetooth/scan"),
    ).timeout(const Duration(seconds: 15));
    if (response.statusCode == 200) {
      return jsonDecode(response.body);
    } else {
      throw Exception("Bluetooth scan failed (${response.statusCode}): ${response.body}");
    }
  }

  // Bluetooth Connect Device
  Future<Map<String, dynamic>> connectBluetoothDevice(String mac) async {
    final response = await http.post(
      Uri.parse("$baseUrl/api/bluetooth/connect"),
      headers: {"Content-Type": "application/json"},
      body: jsonEncode({"mac": mac}),
    ).timeout(const Duration(seconds: 25));
    if (response.statusCode == 200) {
      return jsonDecode(response.body);
    } else {
      throw Exception("Bluetooth connection failed (${response.statusCode}): ${response.body}");
    }
  }

  // Bluetooth Disconnect Device
  Future<Map<String, dynamic>> disconnectBluetoothDevice(String mac) async {
    final response = await http.post(
      Uri.parse("$baseUrl/api/bluetooth/disconnect"),
      headers: {"Content-Type": "application/json"},
      body: jsonEncode({"mac": mac}),
    ).timeout(const Duration(seconds: 15));
    if (response.statusCode == 200) {
      return jsonDecode(response.body);
    } else {
      throw Exception("Bluetooth disconnection failed (${response.statusCode}): ${response.body}");
    }
  }
}
