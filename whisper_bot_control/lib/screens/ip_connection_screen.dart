import 'package:flutter/material.dart';
import 'package:shared_preferences/shared_preferences.dart';
import '../services/bot_api_client.dart';
import 'dashboard_screen.dart';

class IPConnectionScreen extends StatefulWidget {
  const IPConnectionScreen({Key? key}) : super(key: key);

  @override
  State<IPConnectionScreen> createState() => _IPConnectionScreenState();
}

class _IPConnectionScreenState extends State<IPConnectionScreen> {
  final TextEditingController _ipController = TextEditingController(text: '192.168.12.1:5000');
  bool _isLoading = false;
  String? _errorMessage;

  @override
  void initState() {
    super.initState();
    _loadSavedIP();
  }

  Future<void> _loadSavedIP() async {
    final prefs = await SharedPreferences.getInstance();
    final savedIP = prefs.getString('saved_bot_ip');
    if (savedIP != null && savedIP.isNotEmpty) {
      setState(() {
        _ipController.text = savedIP;
      });
    }
  }

  Future<void> _connectToBot() async {
    final ip = _ipController.text.trim();
    if (ip.isEmpty) {
      setState(() {
        _errorMessage = 'IP Address cannot be empty.';
      });
      return;
    }

    setState(() {
      _isLoading = true;
      _errorMessage = null;
    });

    try {
      final client = BotApiClient(ip);
      final status = await client.getStatus();
      
      // Connection successful! Save the IP
      final prefs = await SharedPreferences.getInstance();
      await prefs.setString('saved_bot_ip', ip);
      
      if (!mounted) return;
      
      // Navigate to Dashboard
      Navigator.pushReplacement(
        context,
        MaterialPageRoute(
          builder: (context) => DashboardScreen(
            client: client,
            initialStatus: status,
          ),
        ),
      );
    } catch (e) {
      if (mounted) {
        setState(() {
          _errorMessage = 'Could not connect to WaveGo bot: $e\n\nMake sure your phone is connected to the bot\'s WiFi network (e.g. WAVE_BOT) and the IP address is correct.';
        });
      }
    } finally {
      if (mounted) {
        setState(() {
          _isLoading = false;
        });
      }
    }
  }

  @override
  void dispose() {
    _ipController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Container(
        decoration: const BoxDecoration(
          gradient: LinearGradient(
            begin: Alignment.topLeft,
            end: Alignment.bottomRight,
            colors: [
              Color(0xFF07111F),
              Color(0xFF0E1A2F),
            ],
          ),
        ),
        child: SafeArea(
          child: Center(
            child: SingleChildScrollView(
              padding: const EdgeInsets.symmetric(horizontal: 28.0),
              child: Column(
                mainAxisAlignment: MainAxisAlignment.center,
                crossAxisAlignment: CrossAxisAlignment.center,
                children: [
                  // Logo/Header
                  Container(
                    width: 90,
                    height: 90,
                    decoration: BoxDecoration(
                      borderRadius: BorderRadius.circular(28),
                      gradient: const LinearGradient(
                        colors: [
                          Color(0xFF4DE3B7),
                          Color(0xFF7DD3FC),
                        ],
                      ),
                      boxShadow: [
                        BoxShadow(
                          color: const Color(0xFF4DE3B7).withOpacity(0.3),
                          blurRadius: 20,
                          offset: const Offset(0, 10),
                        ),
                      ],
                    ),
                    alignment: Alignment.center,
                    child: const Text(
                      '🐕',
                      style: TextStyle(fontSize: 48),
                    ),
                  ),
                  const SizedBox(height: 24),
                  const Text(
                    'WAVEGO CONTROLLER',
                    style: TextStyle(
                      color: Colors.white,
                      fontSize: 22,
                      fontWeight: FontWeight.bold,
                      letterSpacing: 1.5,
                    ),
                  ),
                  const SizedBox(height: 8),
                  const Text(
                    'Whisper Bot Mobile Terminal',
                    style: TextStyle(
                      color: Color(0xFF9FB1CE),
                      fontSize: 14,
                    ),
                  ),
                  const SizedBox(height: 48),
                  
                  // Connection Card
                  Container(
                    padding: const EdgeInsets.all(24),
                    decoration: BoxDecoration(
                      color: const Color(0xFF0C1526).withOpacity(0.6),
                      borderRadius: BorderRadius.circular(24),
                      border: Border.all(
                        color: const Color(0xFF9DB1CE).withOpacity(0.15),
                      ),
                    ),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        const Text(
                          'ENTER BOT ADDRESS',
                          style: TextStyle(
                            color: Color(0xFF7DD3FC),
                            fontSize: 12,
                            fontWeight: FontWeight.bold,
                            letterSpacing: 1.0,
                          ),
                        ),
                        const SizedBox(height: 12),
                        TextField(
                          controller: _ipController,
                          style: const TextStyle(color: Colors.white, fontFamily: 'monospace'),
                          decoration: InputDecoration(
                            hintText: 'e.g. 192.168.12.1:5000',
                            hintStyle: TextStyle(color: Colors.white.withOpacity(0.3)),
                            filled: true,
                            fillColor: const Color(0xFF040A12).withOpacity(0.7),
                            border: OutlineInputBorder(
                              borderRadius: BorderRadius.circular(12),
                              borderSide: BorderSide(
                                color: const Color(0xFF9DB1CE).withOpacity(0.18),
                              ),
                            ),
                            focusedBorder: OutlineInputBorder(
                              borderRadius: BorderRadius.circular(12),
                              borderSide: const BorderSide(
                                color: Color(0xFF7DD3FC),
                              ),
                            ),
                            prefixIcon: const Icon(Icons.network_ping, color: Color(0xFF7DD3FC)),
                          ),
                        ),
                        if (_errorMessage != null) ...[
                          const SizedBox(height: 16),
                          Container(
                            padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 12),
                            decoration: BoxDecoration(
                              color: Colors.redAccent.withOpacity(0.15),
                              borderRadius: BorderRadius.circular(12),
                              border: Border.all(color: Colors.redAccent.withOpacity(0.3)),
                            ),
                            child: Row(
                              crossAxisAlignment: CrossAxisAlignment.start,
                              children: [
                                const Icon(Icons.warning_amber_rounded, color: Colors.redAccent, size: 20),
                                const SizedBox(width: 8),
                                Expanded(
                                  child: Text(
                                    _errorMessage!,
                                    style: const TextStyle(color: Color(0xFFFCA5A5), fontSize: 12, height: 1.4),
                                  ),
                                ),
                              ],
                            ),
                          ),
                        ],
                        const SizedBox(height: 24),
                        SizedBox(
                          width: double.infinity,
                          height: 52,
                          child: ElevatedButton(
                            onPressed: _isLoading ? null : _connectToBot,
                            style: ElevatedButton.styleFrom(
                              backgroundColor: const Color(0xFF4DE3B7),
                              foregroundColor: const Color(0xFF07111F),
                              disabledBackgroundColor: const Color(0xFF4DE3B7).withOpacity(0.4),
                              shape: RoundedRectangleBorder(
                                borderRadius: BorderRadius.circular(12),
                              ),
                              elevation: 0,
                            ),
                            child: _isLoading
                                ? const SizedBox(
                                    width: 24,
                                    height: 24,
                                    child: CircularProgressIndicator(
                                      strokeWidth: 2.5,
                                      valueColor: AlwaysStoppedAnimation<Color>(Color(0xFF07111F)),
                                    ),
                                  )
                                : const Text(
                                    'CONNECT TO ROBOT',
                                    style: TextStyle(
                                      fontSize: 15,
                                      fontWeight: FontWeight.bold,
                                      letterSpacing: 1.0,
                                    ),
                                  ),
                          ),
                        ),
                      ],
                    ),
                  ),
                  const SizedBox(height: 32),
                  
                  // Helper tip
                  Container(
                    padding: const EdgeInsets.all(16),
                    decoration: BoxDecoration(
                      color: const Color(0xFF7DD3FC).withOpacity(0.06),
                      borderRadius: BorderRadius.circular(16),
                      border: Border.all(
                        color: const Color(0xFF7DD3FC).withOpacity(0.12),
                      ),
                    ),
                    child: const Row(
                      children: [
                        Icon(Icons.wifi_tethering, color: Color(0xFF7DD3FC), size: 24),
                        SizedBox(width: 12),
                        Expanded(
                          child: Text(
                            'AP Mode: Connect to WiFi network "WAVE_BOT" (password: 12345678) and enter "192.168.12.1:5000"',
                            style: TextStyle(
                              color: Color(0xFF9FB1CE),
                              fontSize: 12,
                              height: 1.4,
                            ),
                          ),
                        ),
                      ],
                    ),
                  ),
                ],
              ),
            ),
          ),
        ),
      ),
    );
  }
}
