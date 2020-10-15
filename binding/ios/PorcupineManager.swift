//
//  Copyright 2018-2020 Picovoice Inc.
//  You may not use this file except in compliance with the license. A copy of the license is located in the "LICENSE"
//  file accompanying this source.
//  Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
//  an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
//  specific language governing permissions and limitations under the License.
//

import AVFoundation
import pv_porcupine

public enum PorcupineManagerError: Error {
    case outOfMemory
    case io
    case invalidArgument
}

public enum PorcupineManagerPermissionError: Error {
    case recordingDenied
}

public struct WakeWordConfiguration {
    let name: String
    let filePath: String
    let sensitivity: Float
    
    /// Initializer for the wake word configuration.
    ///
    /// - Parameters:
    ///   - name: The name to use to help identify this configuration.
    ///   - filePath: Absolute path to keyword file containing hyper-parameters (ppn).
    ///   - sensitivity: Sensitivity parameter. A higher sensitivity value lowers miss rate at the cost of increased
    ///     false alarm rate. For more information regarding this parameter refer to 'include/pv_porcupine.h'.
    public init(name: String, filePath: String, sensitivity: Float) {
        self.name = name
        self.filePath = filePath
        self.sensitivity = sensitivity
    }
}

/// High-level iOS binding for Porcupine wake word engine. It handles recording audio from microphone, processes it in real-time using Porcupine,
/// and notifies the client when any of the given keywords are detected.
public class PorcupineManager {
    
    private var porcupine: OpaquePointer?
    
    private let audioInputEngine: AudioInputEngine
    
    /// Whether current manager is listening to audio input.
    public private(set) var isListening = false
    
    private var shouldBeListening: Bool = false
    
    private var onDetection: ((Int32) -> Void)?
    
    /// Initializer for multiple keywords detection.
    ///
    /// - Parameters:
    ///   - modelFilePath: Absolute path to file containing model parameters.
    ///   - wakeKeywordConfigurations: Keyword configurations to use.
    ///   - onDetection: Detection handler to call after wake word detection. The handler is executed on main thread.
    /// - Throws: PorcupineManagerError
    public init(modelPath: String, keywordPaths: [String], sensitivities: [Float32], onDetection: ((Int32) -> Void)?) throws {
        self.onDetection = onDetection
        
        self.audioInputEngine = AudioInputEngine_AudioQueue()
        
        audioInputEngine.audioInput = { [weak self] audio in
            
            guard let `self` = self else {
                return
            }
            
            var result: Int32 = -1
            
            pv_porcupine_process(self.porcupine, audio, &result)
            if result >= 0 {
                DispatchQueue.main.async {
                    self.onDetection?(result)
                }
            }
        }
        
        let status = pv_porcupine_init(
            modelPath,
            Int32(keywordPaths.count),
            keywordPaths.map { UnsafePointer(strdup($0)) },
            sensitivities,
            &porcupine)
        try checkInitStatus(status)
    }
    
    /// Initializer for single keyword detection.
    ///
    /// - Parameters:
    ///   - modelFilePath: Absolute path to file containing model parameters.
    ///   - wakeKeywordConfiguration: Keyword configuration to use.
    ///   - onDetection: Detection handler to call after wake word detection. The handler is executed on main thread.
    /// - Throws: PorcupineManagerError
    public convenience init(modelPath: String, keywordPath: String, sensitivity: Float32, onDetection: ((Int32) -> Void)?) throws {
        try self.init(modelPath: modelPath, keywordPaths: [keywordPath], sensitivities: [sensitivity], onDetection: onDetection)
    }
    
    deinit {
        if isListening {
            stopListening()
        }
        pv_porcupine_delete(porcupine)
        porcupine = nil
    }
    
    /// Start listening for configured wake words.
    ///
    /// - Throws: AVAudioSession, AVAudioEngine errors. Additionally PorcupineManagerPermissionError if
    ///           microphone permission is not granted.
    public func startListening() throws {
        
        shouldBeListening = true
        
        let audioSession = AVAudioSession.sharedInstance()
        // Only check if it's denied, permission will be automatically asked.
        if audioSession.recordPermission == .denied {
            throw PorcupineManagerPermissionError.recordingDenied
        }
        
        guard !isListening else {
            return
        }
        
        try audioSession.setCategory(AVAudioSession.Category.record)
        try audioSession.setMode(AVAudioSession.Mode.measurement)
        try audioSession.setActive(true, options: .notifyOthersOnDeactivation)
        
        try audioInputEngine.start()
        
        isListening = true
    }
    
    /// Stop listening for wake words.
    public func stopListening() {
        
        shouldBeListening = false
        
        guard isListening else {
            return
        }
        
        audioInputEngine.stop()
        isListening = false
    }
    
    // MARK: - Private
    
    private func checkInitStatus(_ status: pv_status_t) throws {
        switch status {
        case PV_STATUS_IO_ERROR:
            throw PorcupineManagerError.io
        case PV_STATUS_OUT_OF_MEMORY:
            throw PorcupineManagerError.outOfMemory
        case PV_STATUS_INVALID_ARGUMENT:
            throw PorcupineManagerError.invalidArgument
        default:
            return
        }
    }
    
    @objc private func onAudioSessionInterruption(_ notification: Notification) {
        
        guard let userInfo = notification.userInfo,
              let typeValue = userInfo[AVAudioSessionInterruptionTypeKey] as? UInt,
              let type = AVAudioSession.InterruptionType(rawValue: typeValue) else {
            return
        }
        
        if type == .began {
            audioInputEngine.pause()
        } else if type == .ended {
            // Interruption options are ignored. AudioEngine should be restarted
            // unless PorcupineManager is told to stop listening.
            guard let _ = userInfo[AVAudioSessionInterruptionOptionKey] as? UInt else {
                return
            }
            if shouldBeListening {
                audioInputEngine.unpause()
            }
        }
    }
}

private protocol AudioInputEngine: AnyObject {
    
    var audioInput: ((UnsafePointer<Int16>) -> Void)? { get set }
    
    func start() throws
    func stop()
    
    func pause()
    func unpause()
}

private class AudioInputEngine_AudioQueue: AudioInputEngine {
    
    private let numBuffers = 3
    private var audioQueue: AudioQueueRef?
    
    var audioInput: ((UnsafePointer<Int16>) -> Void)?
    
    func start() throws {
        var format = AudioStreamBasicDescription(
            mSampleRate: Float64(pv_sample_rate()),
            mFormatID: kAudioFormatLinearPCM,
            mFormatFlags: kLinearPCMFormatFlagIsSignedInteger | kLinearPCMFormatFlagIsPacked,
            mBytesPerPacket: 2,
            mFramesPerPacket: 1,
            mBytesPerFrame: 2,
            mChannelsPerFrame: 1,
            mBitsPerChannel: 16,
            mReserved: 0)
        let userData = UnsafeMutableRawPointer(Unmanaged.passUnretained(self).toOpaque())
        AudioQueueNewInput(&format, createAudioQueueCallback(), userData, nil, nil, 0, &audioQueue)
        
        guard let queue = audioQueue else {
            return
        }
        
        let bufferSize = UInt32(pv_porcupine_frame_length()) * 2
        for _ in 0..<numBuffers {
            var bufferRef: AudioQueueBufferRef? = nil
            AudioQueueAllocateBuffer(queue, bufferSize, &bufferRef)
            if let buffer = bufferRef {
                AudioQueueEnqueueBuffer(queue, buffer, 0, nil)
            }
        }
        
        AudioQueueStart(queue, nil)
    }
    
    func stop() {
        guard let audioQueue = audioQueue else {
            return
        }
        AudioQueueStop(audioQueue, true)
        AudioQueueDispose(audioQueue, false)
    }
    
    func pause() {
        guard let audioQueue = audioQueue else {
            return
        }
        AudioQueuePause(audioQueue)
    }
    
    func unpause() {
        guard let audioQueue = audioQueue else {
            return
        }
        AudioQueueFlush(audioQueue)
        AudioQueueStart(audioQueue, nil)
    }
    
    private func createAudioQueueCallback() -> AudioQueueInputCallback {
        return { userData, queue, bufferRef, startTimeRef, numPackets, packetDescriptions in
            
            // `self` is passed in as userData in the audio queue callback.
            guard let userData = userData else {
                return
            }
            let `self` = Unmanaged<AudioInputEngine_AudioQueue>.fromOpaque(userData).takeUnretainedValue()
            
            let pcm = bufferRef.pointee.mAudioData.assumingMemoryBound(to: Int16.self)
            
            if let audioInput = self.audioInput {
                audioInput(pcm)
            }
            
            AudioQueueEnqueueBuffer(queue, bufferRef, 0, nil)
        }
    }
}
