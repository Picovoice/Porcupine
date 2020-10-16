# Porcupine Wake Word Engine

Made in Vancouver, Canada by [Picovoice](https://picovoice.ai)

Porcupine is a highly-accurate and lightweight wake word engine. It enables building always-listening voice-enabled
applications. 

Porcupine is:

- using deep neural networks trained in real-world environments.
- compact and computationally-efficient making it perfect for IoT.
- scalable. It can detect multiple always-listening voice commands with no added CPU/memory footprint.
- self-service. Developers can train custom wake phrases using [Picovoice Console](https://picovoice.ai/console/).

## Compatibility

- TBD
- Runs on Linux (x86_64), MacOS (x86_64) and Windows (x86_64)


javac -d build -cp "junit-platform-console-standalone-1.7.0.jar" "src\ai\picovoice\porcupine\*.java"
java -ea "-Djava.library.path=.\jniLibs\"" -jar "junit-platform-console-standalone-1.7.0.jar" -cp build -c ai.picovoice.porcupine.PorcupineTest