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

// +build linux darwin

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
	"unsafe"
)

// private vars
var (
	lib                           = C.dlopen(C.CString(libName), C.RTLD_NOW)
	pv_porcupine_init_ptr         = C.dlsym(lib, C.CString("pv_porcupine_init"))
	pv_porcupine_process_ptr      = C.dlsym(lib, C.CString("pv_porcupine_process"))
	pv_sample_rate_ptr            = C.dlsym(lib, C.CString("pv_sample_rate"))
	pv_porcupine_version_ptr      = C.dlsym(lib, C.CString("pv_porcupine_version"))
	pv_porcupine_frame_length_ptr = C.dlsym(lib, C.CString("pv_porcupine_frame_length"))
	pv_porcupine_delete_ptr       = C.dlsym(lib, C.CString("pv_porcupine_delete"))
)

func (porcupine *Porcupine) nativeInit() int {
	var (
		modelPathC  = C.CString(porcupine.ModelPath)
		numKeywords = len(porcupine.KeywordPaths)
		keywordsC   = make([]*C.char, numKeywords)
	)

	for i, s := range porcupine.KeywordPaths {
		keywordsC[i] = C.CString(s)
	}

	var ret = C.pv_porcupine_init_wrapper(pv_porcupine_init_ptr,
		modelPathC,
		(C.int32_t)(numKeywords),
		(**C.char)(unsafe.Pointer(&keywordsC[0])),
		(*C.float)(unsafe.Pointer(&porcupine.Sensitivities[0])),
		(**C.pv_porcupine_t)(unsafe.Pointer(&porcupine.handle)))

	return int(ret)
}

func (porcupine *Porcupine) nativeDelete() {
	C.pv_porcupine_delete_wrapper(pv_porcupine_delete_ptr,
		(*C.pv_porcupine_t)(unsafe.Pointer(porcupine.handle)))
}

func (porcupine *Porcupine) nativeProcess(pcm []int16) (int, int) {

	var index int32
	var ret = C.pv_porcupine_process_wrapper(pv_porcupine_process_ptr,
		(*C.pv_porcupine_t)(unsafe.Pointer(porcupine.handle)),
		(*C.int16_t)(unsafe.Pointer(&pcm[0])),
		(*C.int32_t)(unsafe.Pointer(&index)))
	return int(ret), int(index)
}

func nativeSampleRate() int {
	return int(C.pv_sample_rate_wrapper(pv_sample_rate_ptr))
}

func nativeFrameLength() int {
	return int(C.pv_porcupine_frame_length_wrapper(pv_porcupine_frame_length_ptr))
}

func nativeVersion() string {
	return C.GoString(C.pv_porcupine_version_wrapper(pv_porcupine_version_ptr))
}
