// Copyright 2021 Picovoice Inc.
//
// You may not use this file except in compliance with the license. A copy of the license is
// located in the "LICENSE" file accompanying this source.
//
// Unless required by applicable law or agreed to in writing, software distributed under the
// License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
// express or implied. See the License for the specific language governing permissions and
// limitations under the License.
//

// Go binding for Porcupine wake word engine. It detects utterances of given keywords within an incoming stream of
// audio in real-time. It processes incoming audio in consecutive frames and for each frame emits the detection result.
// The number of samples per frame can be attained by calling `.FrameLength`. The incoming audio needs to have a
// sample rate equal to `.SampleRate` and be 16-bit linearly-encoded. Porcupine operates on single-channel audio.

package porcupine

import (
	"C"
	"embed"
	"fmt"
	"io/ioutil"
	"log"
	"os"
	"path/filepath"
	"runtime"
	"strings"
)

//go:embed embedded
var embeddedFS embed.FS

// PvStatus type
type PvStatus int

// Possible status return codes from the Porcupine library
const (
	SUCCESS          PvStatus = 0
	OUT_OF_MEMORY    PvStatus = 1
	IO_ERROR         PvStatus = 2
	INVALID_ARGUMENT PvStatus = 3
	STOP_ITERATION   PvStatus = 4
	KEY_ERROR        PvStatus = 5
	INVALID_STATE    PvStatus = 6
)

func pvStatusToString(status PvStatus) string {
	switch status {
	case SUCCESS:
		return "SUCCESS"
	case OUT_OF_MEMORY:
		return "OUT_OF_MEMORY"
	case IO_ERROR:
		return "IO_ERROR"
	case INVALID_ARGUMENT:
		return "INVALID_ARGUMENT"
	case STOP_ITERATION:
		return "STOP_ITERATION"
	case KEY_ERROR:
		return "KEY_ERROR"
	case INVALID_STATE:
		return "INVALID_STATE"
	default:
		return fmt.Sprintf("Unknown error code: %d", status)
	}
}

// BuiltInKeyword Type
type BuiltInKeyword string

// Available built-in wake words constants
const (
	ALEXA       BuiltInKeyword = "alexa"
	AMERICANO   BuiltInKeyword = "americano"
	BLUEBERRY   BuiltInKeyword = "blueberry"
	BUMBLEBEE   BuiltInKeyword = "bumblebee"
	COMPUTER    BuiltInKeyword = "computer"
	GRAPEFRUIT  BuiltInKeyword = "grapefruit"
	GRASSHOPPER BuiltInKeyword = "grasshopper"
	HEY_BARISTA BuiltInKeyword = "hey barista"
	HEY_GOOGLE  BuiltInKeyword = "hey google"
	HEY_SIRI    BuiltInKeyword = "hey siri"
	JARVIS      BuiltInKeyword = "jarvis"
	OK_GOOGLE   BuiltInKeyword = "ok google"
	PICOVOICE   BuiltInKeyword = "picovoice"
	PORCUPINE   BuiltInKeyword = "porcupine"
	TERMINATOR  BuiltInKeyword = "terminator"
)

// List of available built-in wake words
var BuiltInKeywords = []BuiltInKeyword{
	ALEXA, AMERICANO, BLUEBERRY, BUMBLEBEE, COMPUTER, GRAPEFRUIT, GRASSHOPPER, HEY_BARISTA,
	HEY_GOOGLE, HEY_SIRI, JARVIS, OK_GOOGLE, PICOVOICE, PORCUPINE, TERMINATOR,
}

// Checks if a given BuiltInKeyword is valid
func (k BuiltInKeyword) IsValid() bool {
	for _, b := range BuiltInKeywords {
		if k == b {
			return true
		}
	}
	return false
}

// Porcupine struct
type Porcupine struct {
	// handle for porcupine instance in C
	handle uintptr

	// Absolute path to the file containing model parameters.
	ModelPath string

	// Sensitivity values for detecting keywords. Each value should be a number within [0, 1]. A
	// higher sensitivity results in fewer misses at the cost of increasing the false alarm rate.
	Sensitivities []float32

	// List of built-in keywords to use.
	BuiltInKeywords []BuiltInKeyword

	// Absolute paths to keyword model files.
	KeywordPaths []string
}

type nativePorcupineInterface interface {
	nativeInit(*Porcupine)
	nativeProcess(*Porcupine, []int)
	nativeDelete(*Porcupine)
	nativeSampleRate()
	nativeFrameLength()
	nativeVersion()
}
type nativePorcupineType struct{}

// private vars
var (
	osName        = getOS()
	extractionDir = filepath.Join(os.TempDir(), "porcupine")

	defaultModelFile = extractDefaultModel()
	builtinKeywords  = extractKeywordFiles()
	libName          = extractLib()
	nativePorcupine  = nativePorcupineType{}
)

var (
	// Number of audio samples per frame.
	FrameLength = nativePorcupine.nativeFrameLength()

	// Audio sample rate accepted by Picovoice.
	SampleRate = nativePorcupine.nativeSampleRate()

	// Porcupine version
	Version = nativePorcupine.nativeVersion()
)

// Init function for Porcupine. Must be called before attempting process
func (porcupine *Porcupine) Init() (err error) {
	if porcupine.ModelPath == "" {
		porcupine.ModelPath = defaultModelFile
	}

	if _, err := os.Stat(porcupine.ModelPath); os.IsNotExist(err) {
		return fmt.Errorf("%s: Specified model file could not be found at %s", pvStatusToString(INVALID_ARGUMENT), porcupine.ModelPath)
	}

	if porcupine.BuiltInKeywords != nil && len(porcupine.BuiltInKeywords) > 0 {
		for _, keyword := range porcupine.BuiltInKeywords {
			if !keyword.IsValid() {
				return fmt.Errorf("%s: '%s' is not a valid built-in keyword.", pvStatusToString(INVALID_ARGUMENT), keyword)
			}
			keywordStr := string(keyword)
			porcupine.KeywordPaths = append(porcupine.KeywordPaths, builtinKeywords[keywordStr])
		}
	}

	if porcupine.KeywordPaths == nil || len(porcupine.KeywordPaths) == 0 {
		return fmt.Errorf("%s: No valid keywords were provided.", pvStatusToString(INVALID_ARGUMENT))
	}

	for _, k := range porcupine.KeywordPaths {
		if _, err := os.Stat(k); os.IsNotExist(err) {
			return fmt.Errorf("%s: Keyword file could not be found at %s", pvStatusToString(INVALID_ARGUMENT), k)
		}
	}

	if porcupine.Sensitivities == nil {
		porcupine.Sensitivities = make([]float32, len(porcupine.KeywordPaths))
		for i := range porcupine.KeywordPaths {
			porcupine.Sensitivities[i] = 0.5
		}
	} else {
		for _, s := range porcupine.Sensitivities {
			if s < 0 || s > 1 {
				return fmt.Errorf("%s: Sensitivity value of %f is invalid. Must be between [0, 1].",
					pvStatusToString(INVALID_ARGUMENT), s)
			}
		}
	}

	if len(porcupine.KeywordPaths) != len(porcupine.Sensitivities) {
		return fmt.Errorf("%s: Keyword array size (%d) is not the same size as sensitivities array (%d)",
			pvStatusToString(INVALID_ARGUMENT), len(porcupine.KeywordPaths), len(porcupine.Sensitivities))
	}

	ret := nativePorcupine.nativeInit(porcupine)
	if PvStatus(ret) != SUCCESS {
		return fmt.Errorf(": Porcupine returned error %s", pvStatusToString(INVALID_ARGUMENT))
	}

	return nil
}

// Releases resources acquired by Porcupine.
func (porcupine *Porcupine) Delete() error {
	if porcupine.handle == 0 {
		return fmt.Errorf("Porcupine has not been initialized or has already been deleted.")
	}

	nativePorcupine.nativeDelete(porcupine)
	return nil
}

// Processes a frame of the incoming audio stream and emits the detection result.
// Frame of audio The number of samples per frame can be attained by calling
// `.FrameLength`. The incoming audio needs to have a sample rate equal to `.Sample` and be 16-bit
// linearly-encoded. Porcupine operates on single-channel audio.
// Returns a 0 based index if keyword was detected in frame. Returns -1 if no detection was made.
func (porcupine *Porcupine) Process(pcm []int16) (keywordIndex int, err error) {

	if porcupine.handle == 0 {
		return -1, fmt.Errorf("Porcupine has not been initialized or has been deleted.")
	}

	if len(pcm) != FrameLength {
		return -1, fmt.Errorf("Input data frame size (%d) does not match required size of %d", len(pcm), FrameLength)
	}

	// call process
	ret, index := nativePorcupine.nativeProcess(porcupine, pcm)
	if PvStatus(ret) != SUCCESS {
		return -1, fmt.Errorf("Process audio frame failed with PvStatus: %d", ret)
	}

	return index, nil
}

func getOS() string {
	switch o := runtime.GOOS; o {
	case "darwin":
		return "mac"
	case "linux":
		// Detect if we are running on a Raspberry Pi
		// For details, see: https://raspberrypi.stackexchange.com/a/118473/46921
		rel, err := os.ReadFile("/sys/firmware/devicetree/base/model")
		if err != nil {
			return "linux"
		}
		if strings.Contains(strings.ToLower(string(rel)), "raspberry pi") {
			return "raspberry-pi"
		}
		return "linux"
	case "windows":
		return "windows"
	default:
		log.Fatalf("%s is not a supported OS", o)
		return ""
	}
}

func getArch() string {
	switch runtime.GOARCH {
	case "arm":
		return "arm11"
	//TODO Handle more arches
	default:
		return "x86_64"
	}
}

func extractDefaultModel() string {
	modelPath := "embedded/lib/common/porcupine_params.pv"
	return extractFile(modelPath, extractionDir)
}

func extractKeywordFiles() map[string]string {
	keywordDirPath := "embedded/resources/keyword_files/" + osName
	keywordFiles, err := embeddedFS.ReadDir(keywordDirPath)
	if err != nil {
		log.Fatalf("%v", err)
	}

	extractedKeywords := make(map[string]string)
	for _, keywordFile := range keywordFiles {
		keywordPath := keywordDirPath + "/" + keywordFile.Name()
		keywordName := strings.Split(keywordFile.Name(), "_")[0]
		extractedKeywords[keywordName] = extractFile(keywordPath, extractionDir)
	}
	return extractedKeywords
}

func extractLib() string {
	var libPath string
	switch o := runtime.GOOS; o {
	case "darwin":
		libPath = fmt.Sprintf("embedded/lib/%s/x86_64/libpv_porcupine.dylib", osName)
	case "linux":
		osArch := getArch()
		libPath = fmt.Sprintf("embedded/lib/%s/%s/libpv_porcupine.so", osName, osArch)
	case "windows":
		libPath = fmt.Sprintf("embedded/lib/%s/amd64/libpv_porcupine.dll", osName)
	default:
		log.Fatalf("%s is not a supported OS", o)
	}

	return extractFile(libPath, extractionDir)
}

func extractFile(srcFile string, dstDir string) string {
	bytes, readErr := embeddedFS.ReadFile(srcFile)
	if readErr != nil {
		log.Fatalf("%v", readErr)
	}

	extractedFilepath := filepath.Join(dstDir, srcFile)
	os.MkdirAll(filepath.Dir(extractedFilepath), 0777)
	writeErr := ioutil.WriteFile(extractedFilepath, bytes, 0777)
	if writeErr != nil {
		log.Fatalf("%v", writeErr)
	}
	return extractedFilepath
}
