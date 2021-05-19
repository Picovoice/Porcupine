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

/*
#cgo LDFLAGS: -ldl
#include <dlfcn.h>
#include <stdint.h>

#include "pv_porcupine.h"

typedef int32_t (*pv_sample_rate_func)();

int32_t pv_sample_rate_wrapper(void *f) {
     return ((pv_sample_rate_func) f)();
}

typedef int32_t (*pv_porcupine_frame_length_func)();

int32_t pv_porcupine_frame_length_wrapper(void* f) {
     return ((pv_porcupine_frame_length_func) f)();
}

typedef char* (*pv_porcupine_version_func)();

char* pv_porcupine_version_wrapper(void* f) {
     return ((pv_porcupine_version_func) f)();
}

typedef pv_status_t (*pv_porcupine_init_func)(const char *, int32_t, const char * const *, const float *, pv_porcupine_t **);

int32_t pv_porcupine_init_wrapper(void *f, const char *model_path, int32_t num_keywords, const char * const *keyword_paths, const float *sensitivities, pv_porcupine_t **object) {
	return ((pv_porcupine_init_func) f)(model_path, num_keywords, keyword_paths, sensitivities, object);
}

typedef pv_status_t (*pv_porcupine_process_func)(pv_porcupine_t *, const int16_t *, int32_t *);

int32_t pv_porcupine_process_wrapper(void *f, pv_porcupine_t *object, const int16_t *pcm, int32_t *keyword_index) {
	return ((pv_porcupine_process_func) f)(object, pcm, keyword_index);
}

typedef void (*pv_porcupine_delete_func)(pv_porcupine_t *);

void pv_porcupine_delete_wrapper(void *f, pv_porcupine_t *object) {
	return ((pv_porcupine_delete_func) f)(object);
}

*/
import "C"

import (
	"embed"
	"errors"
	"fmt"
	"io/ioutil"
	"log"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"unsafe"
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
	default:
		return "Unknown error"
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
	PICO_CLOCK  BuiltInKeyword = "pico clock"
	PICOVOICE   BuiltInKeyword = "picovoice"
	PORCUPINE   BuiltInKeyword = "porcupine"
	TERMINATOR  BuiltInKeyword = "terminator"
)

// List of available built-in wake words
var BuiltInKeywords = []BuiltInKeyword{
	ALEXA, AMERICANO, BLUEBERRY, BUMBLEBEE, COMPUTER, GRAPEFRUIT, GRASSHOPPER, HEY_BARISTA,
	HEY_GOOGLE, HEY_SIRI, JARVIS, OK_GOOGLE, PICO_CLOCK, PICOVOICE, PORCUPINE, TERMINATOR,
}

// Checks if a given BuiltInKeyword is valid
func (k BuiltInKeyword) IsValid() error {
	for _, b := range BuiltInKeywords {
		if k == b {
			return nil
		}
	}
	return errors.New("Invalid built-in keyword.")
}

// Porcupine struct
type Porcupine struct {
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

// private vars
var (
	osName        = getOS()
	extractionDir = filepath.Join(os.TempDir(), "porcupine")

	defaultModelFile = extractDefaultModel()
	builtinKeywords  = extractKeywordFiles()
	libName          = extractLib()

	lib                           = C.dlopen(C.CString(libName), C.RTLD_NOW)
	pv_porcupine_init_ptr         = C.dlsym(lib, C.CString("pv_porcupine_init"))
	pv_porcupine_process_ptr      = C.dlsym(lib, C.CString("pv_porcupine_process"))
	pv_sample_rate_ptr            = C.dlsym(lib, C.CString("pv_sample_rate"))
	pv_porcupine_version_ptr      = C.dlsym(lib, C.CString("pv_porcupine_version"))
	pv_porcupine_frame_length_ptr = C.dlsym(lib, C.CString("pv_porcupine_frame_length"))
	pv_porcupine_delete_ptr       = C.dlsym(lib, C.CString("pv_porcupine_delete"))
)

var (
	// Number of audio samples per frame.
	FrameLength = int(C.pv_porcupine_frame_length_wrapper(pv_porcupine_frame_length_ptr))

	// Audio sample rate accepted by Picovoice.
	SampleRate = int(C.pv_sample_rate_wrapper(pv_sample_rate_ptr))

	// Porcupine version
	Version = C.GoString(C.pv_porcupine_version_wrapper(pv_porcupine_version_ptr))
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
			if err := keyword.IsValid(); err != nil {
				return fmt.Errorf("%s: '%s' is not a valid built-in keyword.", pvStatusToString(INVALID_ARGUMENT), keyword)
			}
			keywordStr := string(keyword)
			porcupine.KeywordPaths = append(porcupine.KeywordPaths, builtinKeywords[keywordStr])
		}
	}

	if porcupine.KeywordPaths == nil || len(porcupine.KeywordPaths) == 0 {
		return fmt.Errorf("%s: No valid keywords were provided.", pvStatusToString(INVALID_ARGUMENT))
	}

	if porcupine.Sensitivities == nil {
		porcupine.Sensitivities = make([]float32, len(porcupine.KeywordPaths))
		for i := range porcupine.KeywordPaths {
			porcupine.Sensitivities[i] = 0.5
		}
	}

	if len(porcupine.KeywordPaths) != len(porcupine.Sensitivities) {
		return fmt.Errorf("%s: Keyword array size (%d) is not the same size as sensitivities array (%d)",
			pvStatusToString(INVALID_ARGUMENT), len(porcupine.KeywordPaths), len(porcupine.Sensitivities))
	}

	var (
		// modelPathC  = C.CString(porcupine.ModelPath)
		numKeywords = len(porcupine.KeywordPaths)
		keywordsC   = make([]*C.char, numKeywords)
	)

	for i, s := range porcupine.KeywordPaths {
		keywordsC[i] = C.CString(s)
	}

	var ret = C.pv_porcupine_init_wrapper(pv_porcupine_init_ptr,
		C.CString(porcupine.ModelPath),
		(C.int32_t)(numKeywords),
		(**C.char)(unsafe.Pointer(&keywordsC[0])),
		(*C.float)(unsafe.Pointer(&porcupine.Sensitivities[0])),
		(**C.pv_porcupine_t)(unsafe.Pointer(&porcupine.handle)))

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

	C.pv_porcupine_delete_wrapper(pv_porcupine_delete_ptr,
		(*C.pv_porcupine_t)(unsafe.Pointer(porcupine.handle)))
	return nil
}

// Processes a frame of the incoming audio stream and emits the detection result.
// Frame of audio The number of samples per frame can be attained by calling
// `.FrameLength`. The incoming audio needs to have a sample rate equal to `.Sample` and be 16-bit
// linearly-encoded. Porcupine operates on single-channel audio.
// Returns a 0 based index if keyword was detected in frame. Returns -1 if no detection was made.
func (porcupine *Porcupine) Process(pcm []int16) (int, error) {

	if porcupine.handle == 0 {
		return -1, fmt.Errorf("Porcupine has not been initialized or has been deleted.")
	}

	if len(pcm) != FrameLength {
		return -1, fmt.Errorf("Input data frame is wrong size")
	}

	var index int32
	var ret = C.pv_porcupine_process_wrapper(pv_porcupine_process_ptr,
		(*C.pv_porcupine_t)(unsafe.Pointer(porcupine.handle)),
		(*C.int16_t)(unsafe.Pointer(&pcm[0])),
		(*C.int32_t)(unsafe.Pointer(&index)))

	if PvStatus(ret) != SUCCESS {
		return -1, fmt.Errorf("Process audio frame failed with PvStatus: %d", ret)
	}

	return int(index), nil
}

func getOS() string {
	switch os := runtime.GOOS; os {
	case "darwin":
		return "mac"
	case "linux":
		return "linux"
	case "windows":
		return "windows"
	default:
		log.Fatalf("%s is not a supported OS", os)
		return ""
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
	switch os := runtime.GOOS; os {
	case "darwin":
		libPath = fmt.Sprintf("embedded/lib/%s/x86_64/libpv_porcupine.dylib", osName)
	case "linux":
		libPath = fmt.Sprintf("embedded/lib/%s/x86_64/libpv_porcupine.so", osName)
	case "windows":
		libPath = fmt.Sprintf("embedded/lib/%s/amd64/libpv_porcupine.dll", osName)
	default:
		log.Fatalf("%s is not a supported OS", os)
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
