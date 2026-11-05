import 'package:flutter_test/flutter_test.dart';
import 'package:relay_mobile/main.dart';

void main() {
  testWidgets('shows configuration guidance without a dashboard URL', (tester) async {
    await tester.pumpWidget(const RelayMobileApp());

    expect(find.text('Connect Relay'), findsOneWidget);
    expect(find.textContaining('HTTPS dashboard URL'), findsOneWidget);
  });
}
