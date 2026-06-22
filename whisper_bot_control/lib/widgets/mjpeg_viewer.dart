import 'dart:async';
import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;

class MjpegStreamReader extends StatefulWidget {
  final String streamUrl;
  final BoxFit fit;

  const MjpegStreamReader({
    Key? key,
    required this.streamUrl,
    this.fit = BoxFit.contain,
  }) : super(key: key);

  @override
  State<MjpegStreamReader> createState() => _MjpegStreamReaderState();
}

class _MjpegStreamReaderState extends State<MjpegStreamReader> {
  Uint8List? _frameBytes;
  StreamSubscription? _subscription;
  bool _isConnected = false;
  String? _error;

  @override
  void initState() {
    super.initState();
    _connect();
  }

  @override
  void didUpdateWidget(covariant MjpegStreamReader oldWidget) {
    super.didUpdateWidget(oldWidget);
    if (oldWidget.streamUrl != widget.streamUrl) {
      _disconnect();
      _connect();
    }
  }

  @override
  void dispose() {
    _disconnect();
    super.dispose();
  }

  void _disconnect() {
    _subscription?.cancel();
    _subscription = null;
  }

  void _connect() {
    setState(() {
      _isConnected = false;
      _frameBytes = null;
      _error = null;
    });

    try {
      final request = http.Request('GET', Uri.parse(widget.streamUrl));
      request.headers['Cache-Control'] = 'no-store';

      http.Client().send(request).then((response) {
        if (response.statusCode != 200) {
          if (mounted) {
            setState(() {
              _error = 'HTTP ${response.statusCode}';
            });
          }
          return;
        }

        if (mounted) {
          setState(() {
            _isConnected = true;
          });
        }

        List<int> chunkBuffer = [];
        _subscription = response.stream.listen(
          (data) {
            chunkBuffer.addAll(data);
            _processBuffer(chunkBuffer);
          },
          onError: (err) {
            if (mounted) {
              setState(() {
                _error = err.toString();
                _isConnected = false;
              });
            }
          },
          onDone: () {
            if (mounted) {
              setState(() {
                _isConnected = false;
              });
            }
          },
          cancelOnError: true,
        );
      }).catchError((err) {
        if (mounted) {
          setState(() {
            _error = err.toString();
          });
        }
      });
    } catch (e) {
      setState(() {
        _error = e.toString();
      });
    }
  }

  void _processBuffer(List<int> buffer) {
    int? soiIndex;
    int? eoiIndex;

    for (int i = 0; i < buffer.length - 1; i++) {
      if (buffer[i] == 0xFF && buffer[i + 1] == 0xD8) {
        soiIndex = soiIndex ?? i;
      } else if (buffer[i] == 0xFF && buffer[i + 1] == 0xD9) {
        eoiIndex = i + 1;
        break;
      }
    }

    if (soiIndex != null && eoiIndex != null && eoiIndex > soiIndex) {
      final frame = Uint8List.fromList(buffer.sublist(soiIndex, eoiIndex + 1));
      if (mounted) {
        setState(() {
          _frameBytes = frame;
        });
      }
      buffer.removeRange(0, eoiIndex + 1);
    } else if (buffer.length > 3 * 1024 * 1024) {
      // Prevent buffer memory leak if it gets too large without matching SOI/EOI
      buffer.clear();
    }
  }

  @override
  Widget build(BuildContext context) {
    if (_error != null) {
      return Container(
        color: Colors.black87,
        child: Center(
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              const Icon(Icons.error_outline, color: Colors.redAccent, size: 40),
              const SizedBox(height: 10),
              Text(
                'Camera Feed Error: $_error',
                style: const TextStyle(color: Colors.white, fontSize: 13),
              ),
              const SizedBox(height: 12),
              ElevatedButton.icon(
                onPressed: _connect,
                icon: const Icon(Icons.refresh, size: 16),
                label: const Text('Retry Connection'),
                style: ElevatedButton.styleFrom(
                  backgroundColor: const Color(0xFF101c31),
                  foregroundColor: Colors.white,
                ),
              ),
            ],
          ),
        ),
      );
    }

    if (!_isConnected || _frameBytes == null) {
      return Container(
        color: Colors.black87,
        child: const Center(
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              SizedBox(
                width: 24,
                height: 24,
                child: CircularProgressIndicator(
                  strokeWidth: 2,
                  valueColor: AlwaysStoppedAnimation<Color>(Color(0xFF7DD3FC)),
                ),
              ),
              SizedBox(height: 12),
              Text(
                'Connecting to Live Video Feed...',
                style: TextStyle(color: Colors.white70, fontSize: 13),
              ),
            ],
          ),
        ),
      );
    }

    return Image.memory(
      _frameBytes!,
      fit: widget.fit,
      gaplessPlayback: true,
    );
  }
}
