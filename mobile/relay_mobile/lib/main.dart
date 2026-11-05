import 'package:flutter/material.dart';
import 'package:webview_flutter/webview_flutter.dart';

const _fallbackDashboardUrl = 'https://dashboard.example.com';
const _dashboardUrl = String.fromEnvironment(
  'DASHBOARD_URL',
  defaultValue: _fallbackDashboardUrl,
);

void main() => runApp(const RelayMobileApp());

class RelayMobileApp extends StatelessWidget {
  const RelayMobileApp({super.key});

  @override
  Widget build(BuildContext context) => MaterialApp(
        title: 'Relay',
        debugShowCheckedModeBanner: false,
        theme: ThemeData(
          colorScheme: ColorScheme.fromSeed(
            seedColor: const Color(0xff477c62),
            brightness: Brightness.light,
            surface: const Color(0xfff8f5ee),
          ),
          scaffoldBackgroundColor: const Color(0xfff8f5ee),
          useMaterial3: true,
        ),
        home: const DashboardScreen(),
      );
}

class DashboardScreen extends StatefulWidget {
  const DashboardScreen({super.key});

  @override
  State<DashboardScreen> createState() => _DashboardScreenState();
}

class _DashboardScreenState extends State<DashboardScreen> {
  WebViewController? _controller;
  var _loading = true;
  String? _error;

  bool get _isConfigured => _dashboardUrl != _fallbackDashboardUrl;
  Uri get _dashboardUri => Uri.parse(_dashboardUrl);

  @override
  void initState() {
    super.initState();
    if (!_isConfigured) return;
    _controller = WebViewController()
      ..setJavaScriptMode(JavaScriptMode.unrestricted)
      ..setBackgroundColor(const Color(0xfff8f5ee))
      ..setNavigationDelegate(
        NavigationDelegate(
          onPageStarted: (_) => setState(() {
            _loading = true;
            _error = null;
          }),
          onPageFinished: (_) => setState(() => _loading = false),
          onWebResourceError: (error) {
            if (error.isForMainFrame == true) {
              setState(() {
                _loading = false;
                _error = error.description;
              });
            }
          },
          onNavigationRequest: (request) {
            final target = Uri.tryParse(request.url);
            if (target == null || target.host != _dashboardUri.host) {
              _showMessage('Relay keeps navigation inside your approved dashboard.');
              return NavigationDecision.prevent;
            }
            return NavigationDecision.navigate;
          },
        ),
      );
    _controller!.loadRequest(_dashboardUri);
  }

  void _showMessage(String message) {
    if (mounted) ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text(message)));
  }

  Future<void> _reload() async {
    setState(() => _error = null);
    await _controller?.reload();
  }

  @override
  Widget build(BuildContext context) => Scaffold(
        body: SafeArea(
          child: !_isConfigured
              ? const _ConfigurationRequired()
              : Stack(
                  children: [
                    WebViewWidget(controller: _controller!),
                    if (_loading) const _LoadingOverlay(),
                    if (_error != null) _ConnectionError(message: _error!, onRetry: _reload),
                  ],
                ),
        ),
      );
}

class _LoadingOverlay extends StatelessWidget {
  const _LoadingOverlay();

  @override
  Widget build(BuildContext context) => const ColoredBox(
        color: Color(0x99f8f5ee),
        child: Center(
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              CircularProgressIndicator(color: Color(0xff477c62)),
              SizedBox(height: 14),
              Text('Opening Relay…'),
            ],
          ),
        ),
      );
}

class _ConfigurationRequired extends StatelessWidget {
  const _ConfigurationRequired();

  @override
  Widget build(BuildContext context) => const Padding(
        padding: EdgeInsets.all(28),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Icon(Icons.phone_in_talk_outlined, size: 34, color: Color(0xff477c62)),
            SizedBox(height: 18),
            Text('Connect Relay', style: TextStyle(fontSize: 26, fontWeight: FontWeight.w700)),
            SizedBox(height: 10),
            Text(
              'Launch this app with your HTTPS dashboard URL. The URL is deliberately not embedded in the source.',
              style: TextStyle(height: 1.45),
            ),
          ],
        ),
      );
}

class _ConnectionError extends StatelessWidget {
  const _ConnectionError({required this.message, required this.onRetry});

  final String message;
  final Future<void> Function() onRetry;

  @override
  Widget build(BuildContext context) => ColoredBox(
        color: const Color(0xfff8f5ee),
        child: Center(
          child: Padding(
            padding: const EdgeInsets.all(28),
            child: Column(
              mainAxisSize: MainAxisSize.min,
              children: [
                const Icon(Icons.cloud_off_outlined, size: 34, color: Color(0xff8b572f)),
                const SizedBox(height: 14),
                const Text('Relay could not connect', style: TextStyle(fontSize: 20, fontWeight: FontWeight.w700)),
                const SizedBox(height: 8),
                Text(message, textAlign: TextAlign.center),
                const SizedBox(height: 18),
                FilledButton.icon(onPressed: onRetry, icon: const Icon(Icons.refresh), label: const Text('Try again')),
              ],
            ),
          ),
        ),
      );
}
