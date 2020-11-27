﻿/*
    Copyright 2020 Picovoice Inc.

    You may not use this file except in compliance with the license. A copy of the license is located in the "LICENSE"
    file accompanying this source.

    Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
    an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
    specific language governing permissions and limitations under the License.
*/

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;

using Pv;

namespace PorcupineDemo
{
    /// <summary>
    /// File Demo for Porcupine wake word engine. It takes an input audio file and a list of keywords to search for
    /// and prints the where in the file it detects instances of the keywords.
    /// </summary>                
    public class FileDemo
    {

        /// <summary>
        /// Reads through input file and, upon detecting the specified wake word(s), prints the detection timecode and the wake word.
        /// </summary>
        /// <param name="inputAudioPath">Required argument. Absolute path to input audio file.</param>                
        /// <param name="modelPath">Absolute path to the file containing model parameters. If not set it will be set to the default location.</param>
        /// <param name="keywordPaths">Absolute paths to keyword model files. If not set it will be populated from `keywords` argument.</param>              
        /// <param name="sensitivities">
        /// Sensitivities for detecting keywords. Each value should be a number within [0, 1]. A higher sensitivity results in fewer 
        /// misses at the cost of increasing the false alarm rate. If not set 0.5 will be used.
        /// </param>
        /// <param name="keywords">
        /// List of keywords (phrases) for detection. The list of available (default) keywords can be retrieved 
        /// using `Porcupine.KEYWORDS`. If `keyword_paths` is set then this argument will be ignored.
        /// </param>
        public static void RunDemo(string inputAudioPath, string modelPath, List<string> keywordPaths, List<string> keywords, List<float> sensitivities)
        {
            Porcupine porcupine = null;
            try
            {
                // init porcupine wake word engine
                porcupine = Porcupine.Create(modelPath, keywordPaths, keywords, sensitivities);
                
                if (keywords == null) {
                    keywords = keywordPaths.Select(k=>Path.GetFileNameWithoutExtension(k).Split("_")[0]).ToList();
                }

                using (BinaryReader reader = new BinaryReader(File.Open(inputAudioPath, FileMode.Open)))
                {
                    short numChannels;
                    ValidateWavFile(reader, porcupine.SampleRate, 16, out numChannels);

                    // read audio and send frames to porcupine
                    short[] porcupineFrame = new short[porcupine.FrameLength];
                    int frameIndex = 0;
                    long totalSamplesRead = 0;
                    while (reader.BaseStream.Position != reader.BaseStream.Length)
                    {
                        totalSamplesRead++;
                        porcupineFrame[frameIndex++] = reader.ReadInt16();

                        if (frameIndex == porcupineFrame.Length)
                        {
                            int result = porcupine.Process(porcupineFrame);
                            if (result >= 0)
                            {
                                Console.WriteLine($"Detected {keywords[result]} at " +
                                    $"{Math.Round(totalSamplesRead / (double)porcupine.SampleRate, 2)} sec");
                            }
                            frameIndex = 0;
                        }

                        // skip right channel
                        if (numChannels == 2)
                        {
                            reader.ReadInt16();
                        }
                    }
                }
            }
            finally
            {
                porcupine?.Dispose();
            }           
        }


        /// <summary>
        ///  Reads RIFF header of a WAV file and validates its properties against Picovoice audio processing requirements
        /// </summary>
        /// <param name="reader">WAV file stream reader</param>
        /// <param name="requiredSampleRate">Required sample rate in Hz</param>     
        /// <param name="requiredBitDepth">Required number of bits per sample</param>             
        /// <param name="numChannels">Number of channels can be returned by function</param>
        public static void ValidateWavFile(BinaryReader reader, int requiredSampleRate, short requiredBitDepth, out short numChannels) 
        {                        
            byte[] riffHeader = reader?.ReadBytes(44);

            int riff = BitConverter.ToInt32(riffHeader, 0);
            int wave = BitConverter.ToInt32(riffHeader, 8);
            if (riff != BitConverter.ToInt32(Encoding.UTF8.GetBytes("RIFF"), 0) ||
                wave != BitConverter.ToInt32(Encoding.UTF8.GetBytes("WAVE"), 0))
            {
                throw new ArgumentException("input_audio_path", $"Invalid input audio file format. Input file must be a {requiredSampleRate}kHz, 16-bit WAV file.");
            }

            numChannels = BitConverter.ToInt16(riffHeader, 22);
            int sampleRate = BitConverter.ToInt32(riffHeader, 24);
            short bitDepth = BitConverter.ToInt16(riffHeader, 34);
            if (sampleRate != requiredSampleRate || bitDepth != requiredBitDepth)
            {
                throw new ArgumentException("input_audio_path", $"Invalid input audio file format. Input file must be a {requiredSampleRate}Hz, 16-bit WAV file.");
            }

            if (numChannels == 2)
            {
                Console.WriteLine("Picovoice processes single-channel audio but stereo file is provided. Processing left channel only.");
            }
        }

        public static void Main(string[] args)
        {
            AppDomain.CurrentDomain.UnhandledException += OnUnhandledException;
            if (args.Length == 0) 
            {
                Console.WriteLine(HELP_STR);
                Console.ReadKey();
                return;
            }

            List<string> keywords = null;
            List<string> keywordPaths = null;
            List<float> sensitivities = null;
            string modelPath = null;            
            string inputAudioPath = null;            
            bool showHelp = false;

            // parse command line arguments
            int argIndex = 0;
            while (argIndex < args.Length)
            {
                if (args[argIndex] == "--input_audio_path")
                {
                    if (++argIndex < args.Length)
                    {
                        inputAudioPath = args[argIndex++];
                    }
                }
                else if (args[argIndex] == "--keywords")
                {
                    argIndex++;
                    keywords = new List<string>();
                    while (argIndex < args.Length && Porcupine.KEYWORDS.Contains(args[argIndex]))
                    {
                        keywords.Add(args[argIndex++]);
                    }                    
                }
                else if (args[argIndex] == "--keyword_paths")
                {
                    argIndex++;
                    keywordPaths = new List<string>();
                    while (argIndex < args.Length && File.Exists(args[argIndex]))
                    {
                        keywordPaths.Add(args[argIndex++]);
                    }                    
                }
                else if (args[argIndex] == "--model_path")
                {
                    if (++argIndex < args.Length && File.Exists(args[argIndex]))
                    {
                        modelPath = args[argIndex++];
                    }
                }
                else if (args[argIndex] == "--sensitivities")
                {
                    argIndex++;
                    sensitivities = new List<float>();
                    float sensitivity;
                    while (argIndex < args.Length && float.TryParse(args[argIndex], out sensitivity))
                    {
                        sensitivities.Add(sensitivity);
                        argIndex++;
                    }                    
                }                
                else if (args[argIndex] == "-h" || args[argIndex] == "--help")
                {
                    showHelp = true;
                    argIndex++;
                }
                else
                {
                    argIndex++;
                }
            }

            // print help text and exit
            if (showHelp)
            {
                Console.WriteLine(HELP_STR);
                Console.ReadKey();
                return;
            }

            // argument validation
            if (string.IsNullOrEmpty(inputAudioPath))
            {
                throw new ArgumentNullException("input_audio_path");
            }
            if (!File.Exists(inputAudioPath)) 
            {
                throw new ArgumentException($"Audio file at path {inputAudioPath} does not exist");
            }

            modelPath ??= Porcupine.MODEL_PATH;
            
            int numKeywords = keywordPaths == null ?  keywords.Count() : keywordPaths.Count();

            if (sensitivities == null)
            {
                sensitivities = Enumerable.Repeat(0.5f, numKeywords).ToList();
            }

            // run demo with validated arguments
            RunDemo(inputAudioPath, modelPath, keywordPaths, keywords, sensitivities);
        }

        private static void OnUnhandledException(object sender, UnhandledExceptionEventArgs e)
        {
            Console.WriteLine(e.ExceptionObject.ToString());
            Console.ReadKey();
            Environment.Exit(-1);
        }

        private static readonly string HELP_STR = "Available options: \n " +
            $"\t--input_audio_path (required): Absolute path to input audio file.\n" +
            $"\t--keywords: List of default keywords for detection. Available keywords: {string.Join(", ", Porcupine.KEYWORDS)}\n" +
            $"\t--keyword_paths: Absolute paths to keyword model files. If not set it will be populated from `--keywords` argument\n" +
            $"\t--model_path: Absolute path to the file containing model parameters.\n" +
            $"\t--sensitivities: Sensitivities for detecting keywords. Each value should be a number within [0, 1]. A higher \n" +
             "\t\tsensitivity results in fewer misses at the cost of increasing the false alarm rate. If not set 0.5 will be used.\n";
    }
}
