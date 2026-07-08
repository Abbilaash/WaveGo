import 'package:flutter/material.dart';
import 'screens/ip_connection_screen.dart';

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Whisper-bot Controller',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        brightness: Brightness.dark,
        scaffoldBackgroundColor: const Color(0xFF07111F),
        primaryColor: const Color(0xFF4DE3B7),
        colorScheme: const ColorScheme.dark(
          primary: Color(0xFF4DE3B7),
          secondary: Color(0xFF7DD3FC),
          surface: Color(0xFF0C1526),
          background: Color(0xFF07111F),
          error: Colors.redAccent,
        ),
        textTheme: const TextTheme(
          bodyLarge: TextStyle(color: Colors.white),
          bodyMedium: TextStyle(color: Color(0xFF9FB1CE)),
        ),
      ),
      home: const IPConnectionScreen(),
    );
  }
}
