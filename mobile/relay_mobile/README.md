# Relay mobile

This is a thin native Android/iOS shell for the existing mobile dashboard. It
uses the system WebView rather than duplicating the React interface in Flutter.
The WebView allows JavaScript because the dashboard is a React application, but
it blocks top-level navigation away from the configured dashboard host.

## Run on Android

Accept the Android SDK licences once:

```bash
flutter doctor --android-licenses
```

Start the dashboard from the repository root. For an Android emulator, its
special `10.0.2.2` address maps to the host machine:

```bash
cd frontend
npm run dev -- --host 0.0.0.0
```

Then start Flutter in a second terminal:

```bash
cd mobile/relay_mobile
flutter pub get
flutter run --dart-define=DASHBOARD_URL=http://10.0.2.2:5173
```

The debug Android manifest permits HTTP only for this local-emulator workflow.
Use an HTTPS URL for a physical phone, release build, or Play Store build:

```bash
flutter run --dart-define=DASHBOARD_URL=https://dashboard.example.com
```

## Build

```bash
flutter test
flutter build appbundle \
  --dart-define=DASHBOARD_URL=https://dashboard.example.com
```

The URL is a build/run-time setting and is intentionally not hard-coded. The
dashboard itself still needs `VITE_AGENT_API_URL` and its normal authenticated
backend configuration. Do not embed a production API key in the frontend
bundle or Flutter `--dart-define` values.
