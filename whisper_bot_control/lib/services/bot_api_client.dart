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

  // Handwriting Recognition
  Future<Map<String, dynamic>> detectHandwriting() async {
    final response = await http.post(
      Uri.parse("$baseUrl/api/detect_hand_writing"),
    );
    return jsonDecode(response.body);
  }

  // Chatbot Command text submit
  Future<Map<String, dynamic>> sendChatbotCommand(String command) async {
    final response = await http.post(
      Uri.parse("$baseUrl/api/chatbot/command"),
      headers: {"Content-Type": "application/json"},
      body: jsonEncode({"command": command}),
    );
    return jsonDecode(response.body);
  }
}
