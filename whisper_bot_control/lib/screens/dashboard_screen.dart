import 'dart:async';
import 'package:flutter/material.dart';
import '../services/bot_api_client.dart';
import '../widgets/mjpeg_viewer.dart';
import 'ip_connection_screen.dart';

class DashboardScreen extends StatefulWidget {
  final BotApiClient client;
  final Map<String, dynamic> initialStatus;

  const DashboardScreen({
    Key? key,
    required this.client,
    required this.initialStatus,
  }) : super(key: key);

  @override
  State<DashboardScreen> createState() => _DashboardScreenState();
}

class _DashboardScreenState extends State<DashboardScreen> {
  late Map<String, dynamic> _status;
  Timer? _statusTimer;
  double _speed = 100.0;
  bool _isConnecting = true;

  // Face Learning State
  bool _isLearning = false;
  List<String> _capturedImages = [];
  final TextEditingController _faceNameController = TextEditingController();

  // Face Following State
  bool _isFaceFollowing = false;
  final TextEditingController _faceFollowController = TextEditingController();

  // Color Following State
  String _activeColor = 'none';

  // Ball Search State
  bool _isBallSearching = false;

  // Handwriting State
  bool _isOcrScanning = false;
  List<dynamic> _ocrResults = [];

  // Chatbot State
  final List<Map<String, dynamic>> _messages = [
    {
      "sender": "bot",
      "text": "Hello! I am your AI Command Assistant.\n\n🤖 robot control: Start with 'command' (e.g. 'command forward 5')\n💬 Q&A: Ask questions\n📚 Learn: 'train on <statement>'",
      "time": "System"
    }
  ];
  final TextEditingController _chatController = TextEditingController();
  final ScrollController _chatScrollController = ScrollController();
  bool _chatLoading = false;

  @override
  void initState() {
    super.initState();
    _status = widget.initialStatus;
    _isConnecting = false;
    _startStatusPolling();
  }

  void _startStatusPolling() {
    _statusTimer = Timer.periodic(const Duration(seconds: 4), (timer) async {
      try {
        final data = await widget.client.getStatus();
        if (mounted) {
          setState(() {
            _status = data;
            _isConnecting = false;
          });
        }
      } catch (e) {
        if (mounted) {
          setState(() {
            _isConnecting = true;
          });
        }
      }
    });
  }

  @override
  void dispose() {
    _statusTimer?.cancel();
    _faceNameController.dispose();
    _faceFollowController.dispose();
    _chatController.dispose();
    _chatScrollController.dispose();
    super.dispose();
  }

  void _showNotification(String msg, {bool isError = false}) {
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(
        content: Text(msg),
        backgroundColor: isError ? Colors.redAccent : const Color(0xFF4DE3B7),
        duration: const Duration(seconds: 2),
      ),
    );
  }

  // --- API Call wrappers ---

  Future<void> _move(String action) async {
    try {
      await widget.client.sendMove(action, _speed.toInt());
    } catch (e) {
      _showNotification("Move failed: $e", isError: true);
    }
  }

  Future<void> _tilt(String direction, String action) async {
    try {
      await widget.client.sendTilt(direction, action);
    } catch (e) {
      _showNotification("Tilt failed: $e", isError: true);
    }
  }

  Future<void> _triggerPreset(String action) async {
    try {
      await widget.client.sendDefaultAction(action);
      _showNotification("Preset action '$action' triggered!");
    } catch (e) {
      _showNotification("Preset failed: $e", isError: true);
    }
  }

  Future<void> _toggleFaceDetect(bool start) async {
    try {
      final res = await widget.client.toggleFaceDetection(start ? 'start' : 'stop');
      _showNotification("Face Detection: ${res['mode'] ?? 'updated'}");
    } catch (e) {
      _showNotification("Error: $e", isError: true);
    }
  }

  Future<void> _toggleFaceFollow() async {
    final name = _faceFollowController.text.trim();
    if (!_isFaceFollowing && name.isEmpty) {
      _showNotification("Please enter a name to follow.", isError: true);
      return;
    }

    try {
      if (_isFaceFollowing) {
        await widget.client.toggleFaceFollow('stop');
        setState(() {
          _isFaceFollowing = false;
        });
        _showNotification("Face follow stopped.");
      } else {
        await widget.client.toggleFaceFollow('start', name: name);
        setState(() {
          _isFaceFollowing = true;
        });
        _showNotification("Following face: $name");
      }
    } catch (e) {
      _showNotification("Error: $e", isError: true);
    }
  }

  Future<void> _triggerFaceLearn() async {
    setState(() {
      _isLearning = true;
      _capturedImages.clear();
    });

    try {
      final res = await widget.client.captureFaces();
      if (res['success'] == true) {
        setState(() {
          _capturedImages = List<String>.from(res['images'] ?? []);
        });
        _showFaceSaveDialog();
      } else {
        _showNotification("Face capture failed: ${res['error'] ?? 'Unknown error'}", isError: true);
        setState(() {
          _isLearning = false;
        });
      }
    } catch (e) {
      _showNotification("Error: $e", isError: true);
      setState(() {
        _isLearning = false;
      });
    }
  }

  void _showFaceSaveDialog() {
    showDialog(
      context: context,
      barrierDismissible: false,
      builder: (context) {
        return AlertDialog(
          backgroundColor: const Color(0xFF0C1526),
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(20),
            side: const BorderSide(color: Color(0xFF4DE3B7), width: 1),
          ),
          title: const Text('Save Captured Identity', style: TextStyle(color: Colors.white)),
          content: SingleChildScrollView(
            child: Column(
              mainAxisSize: MainAxisSize.min,
              children: [
                Text(
                  'Captured ${_capturedImages.length} training frames. Enter name to save:',
                  style: const TextStyle(color: Color(0xFF9FB1CE)),
                ),
                const SizedBox(height: 16),
                TextField(
                  controller: _faceNameController,
                  style: const TextStyle(color: Colors.white),
                  decoration: const InputDecoration(
                    labelText: 'Name',
                    labelStyle: TextStyle(color: Color(0xFF7DD3FC)),
                    focusedBorder: UnderlineInputBorder(
                      borderSide: BorderSide(color: Color(0xFF7DD3FC)),
                    ),
                  ),
                ),
              ],
            ),
          ),
          actions: [
            TextButton(
              onPressed: () {
                Navigator.pop(context);
                setState(() {
                  _isLearning = false;
                });
              },
              child: const Text('Cancel', style: TextStyle(color: Colors.grey)),
            ),
            ElevatedButton(
              onPressed: () async {
                final name = _faceNameController.text.trim();
                if (name.isEmpty) return;
                
                Navigator.pop(context);
                try {
                  final res = await widget.client.saveFace(name, _capturedImages);
                  if (res['success'] == true) {
                    _showNotification("Successfully learned face for: $name");
                  } else {
                    _showNotification(res['error'] ?? "Failed to save face.", isError: true);
                  }
                } catch (e) {
                  _showNotification("Save error: $e", isError: true);
                } finally {
                  setState(() {
                    _isLearning = false;
                    _faceNameController.clear();
                  });
                }
              },
              style: ElevatedButton.styleFrom(backgroundColor: const Color(0xFF4DE3B7), foregroundColor: const Color(0xFF07111F)),
              child: const Text('Save'),
            )
          ],
        );
      },
    );
  }

  Future<void> _toggleBallSearch() async {
    try {
      if (_isBallSearching) {
        await widget.client.toggleBallSearch('stop');
        setState(() {
          _isBallSearching = false;
        });
        _showNotification("Ball search deactivated.");
      } else {
        await widget.client.toggleBallSearch('start');
        setState(() {
          _isBallSearching = true;
        });
        _showNotification("Ball search started.");
      }
    } catch (e) {
      _showNotification("Error: $e", isError: true);
    }
  }

  Future<void> _setColorFollow(String color) async {
    try {
      final res = await widget.client.followColor(color);
      setState(() {
        _activeColor = color;
      });
      _showNotification("Color Tracking: ${res['status'] ?? 'Updated'}");
    } catch (e) {
      _showNotification("Error: $e", isError: true);
    }
  }

  Future<void> _runOcr() async {
    setState(() {
      _isOcrScanning = true;
      _ocrResults.clear();
    });

    try {
      final res = await widget.client.detectHandwriting();
      if (res['success'] == true) {
        setState(() {
          _ocrResults = res['results'] ?? [];
        });
        _showNotification("Handwriting scanned successfully!");
      } else {
        _showNotification(res['error'] ?? "OCR failed.", isError: true);
      }
    } catch (e) {
      _showNotification("Error: $e", isError: true);
    } finally {
      setState(() {
        _isOcrScanning = false;
      });
    }
  }

  Future<void> _sendMessage() async {
    final text = _chatController.text.trim();
    if (text.isEmpty) return;

    _chatController.clear();
    setState(() {
      _messages.add({
        "sender": "user",
        "text": text,
        "time": "${DateTime.now().hour.toString().padLeft(2, '0')}:${DateTime.now().minute.toString().padLeft(2, '0')}"
      });
      _chatLoading = true;
    });
    
    _scrollToBottom();

    try {
      final res = await widget.client.sendChatbotCommand(text);
      if (mounted) {
        setState(() {
          _messages.add({
            "sender": "bot",
            "text": res['action_taken'] ?? "No response",
            "time": "Model"
          });
        });
        _scrollToBottom();
      }
    } catch (e) {
      if (mounted) {
        setState(() {
          _messages.add({
            "sender": "bot",
            "text": "Failed to reach AI model: $e",
            "time": "System Error"
          });
        });
        _scrollToBottom();
      }
    } finally {
      if (mounted) {
        setState(() {
          _chatLoading = false;
        });
      }
    }
  }

  void _scrollToBottom() {
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (_chatScrollController.hasClients) {
        _chatScrollController.animateTo(
          _chatScrollController.position.maxScrollExtent,
          duration: const Duration(milliseconds: 300),
          curve: Curves.easeOut,
        );
      }
    });
  }

  // --- UI Components ---

  Widget _buildStatusBadge() {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
      decoration: BoxDecoration(
        color: const Color(0xFF0C1526).withOpacity(0.6),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: const Color(0xFF9DB1CE).withOpacity(0.12)),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Container(
            width: 8,
            height: 8,
            decoration: BoxDecoration(
              color: _isConnecting ? Colors.redAccent : const Color(0xFF4DE3B7),
              shape: BoxShape.circle,
              boxShadow: [
                BoxShadow(
                  color: (_isConnecting ? Colors.redAccent : const Color(0xFF4DE3B7)).withOpacity(0.4),
                  blurRadius: 6,
                  spreadRadius: 2,
                ),
              ],
            ),
          ),
          const SizedBox(width: 8),
          Text(
            _isConnecting ? 'Offline' : 'Connected',
            style: TextStyle(
              color: _isConnecting ? Colors.redAccent : const Color(0xFF4DE3B7),
              fontSize: 12,
              fontWeight: FontWeight.bold,
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildHUDCard(String label, String value) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 12),
      decoration: BoxDecoration(
        color: const Color(0xFF0C1526).withOpacity(0.5),
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: const Color(0xFF9DB1CE).withOpacity(0.12)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            label,
            style: const TextStyle(color: Color(0xFF9FB1CE), fontSize: 10, fontWeight: FontWeight.bold),
          ),
          const SizedBox(height: 6),
          Text(
            value,
            style: const TextStyle(color: Colors.white, fontSize: 13, fontWeight: FontWeight.bold),
          ),
        ],
      ),
    );
  }

  // D-pad Directional Button
  Widget _buildDpadButton({
    required IconData icon,
    required VoidCallback onTapDown,
    required VoidCallback onTapUp,
    Color color = const Color(0xFF7DD3FC),
  }) {
    return GestureDetector(
      onTapDown: (_) => onTapDown(),
      onTapUp: (_) => onTapUp(),
      onTapCancel: () => onTapUp(),
      child: Container(
        width: 60,
        height: 60,
        decoration: BoxDecoration(
          color: const Color(0xFF0C1526).withOpacity(0.8),
          borderRadius: BorderRadius.circular(14),
          border: Border.all(color: color.withOpacity(0.3)),
          boxShadow: [
            BoxShadow(
              color: color.withOpacity(0.05),
              blurRadius: 8,
              offset: const Offset(0, 4),
            )
          ],
        ),
        child: Icon(icon, color: color, size: 28),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    final streamUrl = "${widget.client.baseUrl}/video_feed";
    
    return DefaultTabController(
      length: 3,
      child: Scaffold(
        backgroundColor: const Color(0xFF07111F),
        appBar: AppBar(
          backgroundColor: const Color(0xFF0C1526),
          title: const Text(
            '🐕 WAVEGO',
            style: TextStyle(fontWeight: FontWeight.bold, letterSpacing: 1.0, fontSize: 16),
          ),
          actions: [
            _buildStatusBadge(),
            IconButton(
              icon: const Icon(Icons.logout, size: 20),
              onPressed: () {
                Navigator.pushReplacement(
                  context,
                  MaterialPageRoute(builder: (context) => const IPConnectionScreen()),
                );
              },
            ),
          ],
          bottom: const TabBar(
            indicatorColor: Color(0xFF4DE3B7),
            labelColor: Color(0xFF4DE3B7),
            unselectedLabelColor: Color(0xFF9FB1CE),
            labelStyle: TextStyle(fontWeight: FontWeight.bold, letterSpacing: 0.5, fontSize: 13),
            tabs: [
              Tab(icon: Icon(Icons.gamepad), text: 'Control'),
              Tab(icon: Icon(Icons.remove_red_eye), text: 'Vision'),
              Tab(icon: Icon(Icons.psychology), text: 'Assistant'),
            ],
          ),
        ),
        body: TabBarView(
          children: [
            // Tab 1: Control Room
            SingleChildScrollView(
              padding: const EdgeInsets.all(16.0),
              child: Column(
                children: [
                  // HUD Telemetry
                  GridView.count(
                    crossAxisCount: 3,
                    shrinkWrap: true,
                    physics: const NeverScrollableScrollPhysics(),
                    crossAxisSpacing: 10,
                    childAspectRatio: 2.2,
                    children: [
                      _buildHUDCard('CPU TEMP', _isConnecting ? 'unknown' : '${_status["cpu_temp"] ?? "unknown"}'),
                      _buildHUDCard('RAM USE', _isConnecting ? 'unknown' : '${_status["ram_info"] ?? "unknown"}%'),
                      _buildHUDCard('CPU USE', _isConnecting ? 'unknown' : '${_status["cpu_use"] ?? "unknown"}%'),
                    ],
                  ),
                  const SizedBox(height: 16),

                  // Camera Feed
                  AspectRatio(
                    aspectRatio: 16 / 9,
                    child: Container(
                      clipBehavior: Clip.antiAlias,
                      decoration: BoxDecoration(
                        borderRadius: BorderRadius.circular(16),
                        border: Border.all(color: const Color(0xFF9DB1CE).withOpacity(0.15)),
                      ),
                      child: MjpegStreamReader(streamUrl: streamUrl),
                    ),
                  ),
                  const SizedBox(height: 20),

                  // D-pads & Joysticks
                  Row(
                    mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      // Movement Controls
                      Column(
                        children: [
                          const Text(
                            'MOVE ROBOT',
                            style: TextStyle(color: Color(0xFF4DE3B7), fontWeight: FontWeight.bold, fontSize: 12),
                          ),
                          const SizedBox(height: 12),
                          SizedBox(
                            width: 190,
                            height: 190,
                            child: Stack(
                              children: [
                                Align(
                                  alignment: Alignment.topCenter,
                                  child: _buildDpadButton(
                                    icon: Icons.arrow_upward,
                                    onTapDown: () => _move('forward'),
                                    onTapUp: () => _move('stop'),
                                    color: const Color(0xFF4DE3B7),
                                  ),
                                ),
                                Align(
                                  alignment: Alignment.bottomCenter,
                                  child: _buildDpadButton(
                                    icon: Icons.arrow_downward,
                                    onTapDown: () => _move('backward'),
                                    onTapUp: () => _move('stop'),
                                    color: const Color(0xFF4DE3B7),
                                  ),
                                ),
                                Align(
                                  alignment: Alignment.centerLeft,
                                  child: _buildDpadButton(
                                    icon: Icons.arrow_back,
                                    onTapDown: () => _move('left'),
                                    onTapUp: () => _move('stop'),
                                    color: const Color(0xFF4DE3B7),
                                  ),
                                ),
                                Align(
                                  alignment: Alignment.centerRight,
                                  child: _buildDpadButton(
                                    icon: Icons.arrow_forward,
                                    onTapDown: () => _move('right'),
                                    onTapUp: () => _move('stop'),
                                    color: const Color(0xFF4DE3B7),
                                  ),
                                ),
                                Align(
                                  alignment: Alignment.center,
                                  child: GestureDetector(
                                    onTap: () => _move('stop'),
                                    child: Container(
                                      width: 58,
                                      height: 58,
                                      decoration: BoxDecoration(
                                        color: Colors.redAccent.withOpacity(0.18),
                                        shape: BoxShape.circle,
                                        border: Border.all(color: Colors.redAccent.withOpacity(0.5)),
                                      ),
                                      alignment: Alignment.center,
                                      child: const Text(
                                        'STOP',
                                        style: TextStyle(
                                          color: Colors.redAccent,
                                          fontWeight: FontWeight.bold,
                                          fontSize: 12,
                                        ),
                                      ),
                                    ),
                                  ),
                                ),
                              ],
                            ),
                          ),
                        ],
                      ),

                      // Tilt Controls
                      Column(
                        children: [
                          const Text(
                            'TILT CAMERA',
                            style: TextStyle(color: Color(0xFF7DD3FC), fontWeight: FontWeight.bold, fontSize: 12),
                          ),
                          const SizedBox(height: 12),
                          SizedBox(
                            width: 130,
                            height: 190,
                            child: Column(
                              mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                              children: [
                                _buildDpadButton(
                                  icon: Icons.keyboard_arrow_up,
                                  onTapDown: () => _tilt('up', 'start'),
                                  onTapUp: () => _tilt('up', 'stop'),
                                ),
                                Row(
                                  mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                                  children: [
                                    _buildDpadButton(
                                      icon: Icons.keyboard_arrow_left,
                                      onTapDown: () => _tilt('left', 'start'),
                                      onTapUp: () => _tilt('left', 'stop'),
                                    ),
                                    _buildDpadButton(
                                      icon: Icons.keyboard_arrow_right,
                                      onTapDown: () => _tilt('right', 'start'),
                                      onTapUp: () => _tilt('right', 'stop'),
                                    ),
                                  ],
                                ),
                                _buildDpadButton(
                                  icon: Icons.keyboard_arrow_down,
                                  onTapDown: () => _tilt('down', 'start'),
                                  onTapUp: () => _tilt('down', 'stop'),
                                ),
                              ],
                            ),
                          ),
                        ],
                      ),
                    ],
                  ),
                  const SizedBox(height: 16),

                  // Speed Control Slider
                  Container(
                    padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 10),
                    decoration: BoxDecoration(
                      color: const Color(0xFF0C1526).withOpacity(0.6),
                      borderRadius: BorderRadius.circular(16),
                      border: Border.all(color: const Color(0xFF9DB1CE).withOpacity(0.12)),
                    ),
                    child: Column(
                      children: [
                        Row(
                          mainAxisAlignment: MainAxisAlignment.spaceBetween,
                          children: [
                            const Text('Walk Speed', style: TextStyle(color: Color(0xFF9FB1CE), fontSize: 13)),
                            Text('${_speed.toInt()}%', style: const TextStyle(color: Color(0xFF4DE3B7), fontWeight: FontWeight.bold, fontSize: 13)),
                          ],
                        ),
                        Slider(
                          value: _speed,
                          min: 10.0,
                          max: 100.0,
                          activeColor: const Color(0xFF4DE3B7),
                          inactiveColor: const Color(0xFF07111F),
                          onChanged: (val) {
                            setState(() {
                              _speed = val;
                            });
                          },
                        ),
                      ],
                    ),
                  ),
                  const SizedBox(height: 16),

                  // Presets Row
                  Container(
                    padding: const EdgeInsets.all(14),
                    decoration: BoxDecoration(
                      color: const Color(0xFF0C1526).withOpacity(0.6),
                      borderRadius: BorderRadius.circular(16),
                      border: Border.all(color: const Color(0xFF9DB1CE).withOpacity(0.12)),
                    ),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        const Text('DEFAULT PRESETS', style: TextStyle(color: Color(0xFF9FB1CE), fontWeight: FontWeight.bold, fontSize: 11, letterSpacing: 0.5)),
                        const SizedBox(height: 12),
                        Row(
                          children: [
                            Expanded(
                              child: ElevatedButton(
                                onPressed: () => _triggerPreset('steady'),
                                style: ElevatedButton.styleFrom(backgroundColor: const Color(0xFF040A12), foregroundColor: Colors.white, shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(10))),
                                child: const Text('Steady', style: TextStyle(fontSize: 12)),
                              ),
                            ),
                            const SizedBox(width: 8),
                            Expanded(
                              child: ElevatedButton(
                                onPressed: () => _triggerPreset('jump'),
                                style: ElevatedButton.styleFrom(backgroundColor: const Color(0xFF040A12), foregroundColor: Colors.white, shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(10))),
                                child: const Text('Jump', style: TextStyle(fontSize: 12)),
                              ),
                            ),
                            const SizedBox(width: 8),
                            Expanded(
                              child: ElevatedButton(
                                onPressed: () => _triggerPreset('handshake'),
                                style: ElevatedButton.styleFrom(backgroundColor: const Color(0xFF040A12), foregroundColor: Colors.white, shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(10))),
                                child: const Text('Handshake', style: TextStyle(fontSize: 12)),
                              ),
                            ),
                          ],
                        )
                      ],
                    ),
                  ),
                ],
              ),
            ),

            // Tab 2: Computer Vision Hub
            SingleChildScrollView(
              padding: const EdgeInsets.all(16.0),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  // Face Detection Module
                  Container(
                    padding: const EdgeInsets.all(18),
                    decoration: BoxDecoration(
                      color: const Color(0xFF0C1526).withOpacity(0.6),
                      borderRadius: BorderRadius.circular(18),
                      border: Border.all(color: const Color(0xFF9DB1CE).withOpacity(0.12)),
                    ),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        const Text('FACE RECOGNITION WIZARD', style: TextStyle(color: Color(0xFF4DE3B7), fontWeight: FontWeight.bold, fontSize: 12)),
                        const SizedBox(height: 16),
                        Row(
                          children: [
                            Expanded(
                              child: ElevatedButton(
                                onPressed: _isLearning ? null : _triggerFaceLearn,
                                style: ElevatedButton.styleFrom(backgroundColor: const Color(0xFF4DE3B7), foregroundColor: const Color(0xFF07111F), shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(10))),
                                child: _isLearning
                                  ? const SizedBox(width: 16, height: 16, child: CircularProgressIndicator(strokeWidth: 2, valueColor: AlwaysStoppedAnimation<Color>(Colors.white)))
                                  : const Text('Learn Identity'),
                              ),
                            ),
                          ],
                        ),
                        const SizedBox(height: 12),
                        Row(
                          children: [
                            Expanded(
                              child: OutlinedButton(
                                onPressed: () => _toggleFaceDetect(true),
                                style: OutlinedButton.styleFrom(side: const BorderSide(color: Color(0xFF7DD3FC)), foregroundColor: const Color(0xFF7DD3FC), shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(10))),
                                child: const Text('Detect On'),
                              ),
                            ),
                            const SizedBox(width: 10),
                            Expanded(
                              child: OutlinedButton(
                                onPressed: () => _toggleFaceDetect(false),
                                style: OutlinedButton.styleFrom(side: const BorderSide(color: Colors.grey), foregroundColor: Colors.grey, shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(10))),
                                child: const Text('Detect Off'),
                              ),
                            ),
                          ],
                        ),
                        const SizedBox(height: 16),
                        const Divider(color: Colors.white10),
                        const SizedBox(height: 12),
                        const Text('Face Following', style: TextStyle(color: Colors.white, fontSize: 13)),
                        const SizedBox(height: 8),
                        Row(
                          children: [
                            Expanded(
                              child: TextField(
                                controller: _faceFollowController,
                                style: const TextStyle(color: Colors.white, fontSize: 13),
                                decoration: InputDecoration(
                                  hintText: 'Enter target identity...',
                                  hintStyle: const TextStyle(color: Colors.white24),
                                  filled: true,
                                  fillColor: const Color(0xFF040A12),
                                  border: OutlineInputBorder(borderRadius: BorderRadius.circular(10), borderSide: BorderSide.none),
                                  contentPadding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
                                ),
                              ),
                            ),
                            const SizedBox(width: 10),
                            ElevatedButton(
                              onPressed: _toggleFaceFollow,
                              style: ElevatedButton.styleFrom(
                                backgroundColor: _isFaceFollowing ? Colors.redAccent : const Color(0xFF7DD3FC),
                                foregroundColor: const Color(0xFF07111F),
                                shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(10)),
                              ),
                              child: Text(_isFaceFollowing ? 'Stop' : 'Follow'),
                            ),
                          ],
                        ),
                      ],
                    ),
                  ),
                  const SizedBox(height: 16),

                  // Ball Search and Color Follow Module
                  Container(
                    padding: const EdgeInsets.all(18),
                    decoration: BoxDecoration(
                      color: const Color(0xFF0C1526).withOpacity(0.6),
                      borderRadius: BorderRadius.circular(18),
                      border: Border.all(color: const Color(0xFF9DB1CE).withOpacity(0.12)),
                    ),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        const Text('TARGET COLOR TRACKING', style: TextStyle(color: Color(0xFF4DE3B7), fontWeight: FontWeight.bold, fontSize: 12)),
                        const SizedBox(height: 16),
                        Row(
                          children: [
                            Expanded(
                              child: ElevatedButton(
                                onPressed: _toggleBallSearch,
                                style: ElevatedButton.styleFrom(
                                  backgroundColor: _isBallSearching ? Colors.redAccent : const Color(0xFF4DE3B7),
                                  foregroundColor: const Color(0xFF07111F),
                                  shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(10)),
                                ),
                                child: Text(_isBallSearching ? 'Stop Ball Search' : '🔍 Find Green Ball'),
                              ),
                            ),
                          ],
                        ),
                        const SizedBox(height: 16),
                        const Divider(color: Colors.white10),
                        const SizedBox(height: 12),
                        const Text('Follow Custom Color Filters', style: TextStyle(color: Colors.white, fontSize: 13)),
                        const SizedBox(height: 12),
                        Row(
                          mainAxisAlignment: MainAxisAlignment.spaceBetween,
                          children: [
                            ElevatedButton(
                              onPressed: () => _setColorFollow('green'),
                              style: ElevatedButton.styleFrom(
                                backgroundColor: _activeColor == 'green' ? const Color(0xFF10b981) : const Color(0xFF040A12),
                                foregroundColor: Colors.white,
                              ),
                              child: const Text('Green'),
                            ),
                            ElevatedButton(
                              onPressed: () => _setColorFollow('blue'),
                              style: ElevatedButton.styleFrom(
                                backgroundColor: _activeColor == 'blue' ? const Color(0xFF3b82f6) : const Color(0xFF040A12),
                                foregroundColor: Colors.white,
                              ),
                              child: const Text('Blue'),
                            ),
                            ElevatedButton(
                              onPressed: () => _setColorFollow('red'),
                              style: ElevatedButton.styleFrom(
                                backgroundColor: _activeColor == 'red' ? const Color(0xFFef4444) : const Color(0xFF040A12),
                                foregroundColor: Colors.white,
                              ),
                              child: const Text('Red'),
                            ),
                            ElevatedButton(
                              onPressed: () => _setColorFollow('none'),
                              style: ElevatedButton.styleFrom(
                                backgroundColor: const Color(0xFF1F2937),
                                foregroundColor: Colors.white,
                              ),
                              child: const Text('Stop'),
                            ),
                          ],
                        ),
                      ],
                    ),
                  ),
                  const SizedBox(height: 16),

                  // Handwriting Recognition Module
                  Container(
                    padding: const EdgeInsets.all(18),
                    decoration: BoxDecoration(
                      color: const Color(0xFF0C1526).withOpacity(0.6),
                      borderRadius: BorderRadius.circular(18),
                      border: Border.all(color: const Color(0xFF9DB1CE).withOpacity(0.12)),
                    ),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        const Text('HANDWRITING OCR SCANNER', style: TextStyle(color: Color(0xFF4DE3B7), fontWeight: FontWeight.bold, fontSize: 12)),
                        const SizedBox(height: 16),
                        SizedBox(
                          width: double.infinity,
                          child: ElevatedButton.icon(
                            onPressed: _isOcrScanning ? null : _runOcr,
                            icon: const Icon(Icons.document_scanner),
                            label: const Text('Scan Handwriting'),
                            style: ElevatedButton.styleFrom(
                              backgroundColor: const Color(0xFF7DD3FC),
                              foregroundColor: const Color(0xFF07111F),
                              shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(10)),
                            ),
                          ),
                        ),
                        if (_isOcrScanning) ...[
                          const SizedBox(height: 14),
                          const Center(child: CircularProgressIndicator(strokeWidth: 2)),
                        ],
                        if (_ocrResults.isNotEmpty) ...[
                          const SizedBox(height: 16),
                          Container(
                            padding: const EdgeInsets.all(12),
                            decoration: BoxDecoration(
                              color: const Color(0xFF4DE3B7).withOpacity(0.10),
                              border: Border.all(color: const Color(0xFF4DE3B7).withOpacity(0.3)),
                              borderRadius: BorderRadius.circular(10),
                            ),
                            child: Column(
                              crossAxisAlignment: CrossAxisAlignment.start,
                              children: [
                                const Text('OCR Results:', style: TextStyle(color: Color(0xFF4DE3B7), fontWeight: FontWeight.bold, fontSize: 11)),
                                const SizedBox(height: 8),
                                ..._ocrResults.map((res) {
                                  final text = res[0] ?? '';
                                  final conf = (res[1] ?? 0.0) * 100;
                                  return Padding(
                                    padding: const EdgeInsets.symmetric(vertical: 2.0),
                                    child: Text(
                                      '- "$text" (Confidence: ${conf.toStringAsFixed(1)}%)',
                                      style: const TextStyle(color: Colors.white, fontSize: 12),
                                    ),
                                  );
                                }).toList(),
                              ],
                            ),
                          )
                        ],
                      ],
                    ),
                  ),
                ],
              ),
            ),

            // Tab 3: AI Assistant Terminal
            Column(
              children: [
                // Chat Message List
                Expanded(
                  child: ListView.builder(
                    controller: _chatScrollController,
                    padding: const EdgeInsets.all(16),
                    itemCount: _messages.length,
                    itemBuilder: (context, idx) {
                      final msg = _messages[idx];
                      final isUser = msg["sender"] == "user";
                      return Align(
                        alignment: isUser ? Alignment.centerRight : Alignment.centerLeft,
                        child: Container(
                          margin: const EdgeInsets.symmetric(vertical: 6),
                          padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 10),
                          constraints: BoxConstraints(maxWidth: MediaQuery.of(context).size.width * 0.8),
                          decoration: BoxDecoration(
                            gradient: isUser
                              ? const LinearGradient(colors: [Color(0xFF0C2B4E), Color(0xFF0B1F37)])
                              : const LinearGradient(colors: [Color(0xFF0E322A), Color(0xFF081C17)]),
                            borderRadius: BorderRadius.only(
                              topLeft: const Radius.circular(16),
                              topRight: const Radius.circular(16),
                              bottomLeft: isUser ? const Radius.circular(16) : const Radius.circular(2),
                              bottomRight: isUser ? const Radius.circular(2) : const Radius.circular(16),
                            ),
                            border: Border.all(
                              color: isUser
                                ? const Color(0xFF7DD3FC).withOpacity(0.2)
                                : const Color(0xFF4DE3B7).withOpacity(0.2),
                            ),
                          ),
                          child: Column(
                            crossAxisAlignment: CrossAxisAlignment.start,
                            children: [
                              Text(
                                msg["text"] ?? '',
                                style: const TextStyle(color: Colors.white, fontSize: 13, height: 1.4),
                              ),
                              const SizedBox(height: 6),
                              Text(
                                msg["time"] ?? '',
                                style: const TextStyle(color: Colors.white38, fontSize: 9),
                              ),
                            ],
                          ),
                        ),
                      );
                    },
                  ),
                ),
                
                if (_chatLoading) ...[
                  const LinearProgressIndicator(minHeight: 2, valueColor: AlwaysStoppedAnimation<Color>(Color(0xFF4DE3B7)), backgroundColor: Colors.transparent),
                ],

                // Input Bar
                Container(
                  padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
                  decoration: const BoxDecoration(
                    color: Color(0xFF0C1526),
                    border: Border(top: BorderSide(color: Colors.white10)),
                  ),
                  child: Row(
                    children: [
                      Expanded(
                        child: TextField(
                          controller: _chatController,
                          style: const TextStyle(color: Colors.white, fontSize: 14),
                          decoration: InputDecoration(
                            hintText: 'Enter AI command assistant request...',
                            hintStyle: const TextStyle(color: Colors.white30, fontSize: 13),
                            border: OutlineInputBorder(borderRadius: BorderRadius.circular(12), borderSide: BorderSide.none),
                            filled: true,
                            fillColor: const Color(0xFF040A12),
                            contentPadding: const EdgeInsets.symmetric(horizontal: 14, vertical: 10),
                          ),
                          onSubmitted: (_) => _sendMessage(),
                        ),
                      ),
                      const SizedBox(width: 8),
                      IconButton(
                        icon: const Icon(Icons.send, color: Color(0xFF4DE3B7)),
                        onPressed: _sendMessage,
                      ),
                    ],
                  ),
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }
}
