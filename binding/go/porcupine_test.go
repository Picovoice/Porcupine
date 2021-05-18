package porcupine

import (
	"io/ioutil"
	"math"
	"path/filepath"
	"testing"
)

func TestProcess(t *testing.T) {

	test_file, _ := filepath.Abs("../../resources/audio_samples/porcupine.wav")

	p := Porcupine{BuiltInKeywords: []BuiltInKeyword{PORCUPINE}}
	status, err := p.Init()
	if err != nil {
		t.Fatalf("%v", err)
	}
	if status != SUCCESS {
		t.Fatalf("Porcupine failed to init with PvStatus %d", status)
	}

	t.Logf("Porcupine Version: %s", Version)
	t.Logf("Frame Length: %d", FrameLength)
	t.Logf("Samples Rate: %d", SampleRate)

	data, err := ioutil.ReadFile(test_file)
	if err != nil {
		t.Fatalf("Could not read test file: %v", err)
	}
	data = data[44:] // skip header

	frameLenBytes := FrameLength * 2
	frameCount := int(math.Floor(float64(len(data)) / float64(frameLenBytes)))
	for i := 0; i < frameCount; i++ {
		start := i * frameLenBytes
		count := frameLenBytes

		frame := data[start : start+count]
		result, err := p.Process(frame)
		if err != nil {
			t.Fatalf("Could not read test file: %v", err)
		}
		if result >= 0 {
			t.Logf("Keyword triggered at %f", float64(i*FrameLength)/float64(SampleRate))
		}
	}

	delErr := p.Delete()
	if delErr != nil {
		t.Fatalf("%v", delErr)
	}
}

func TestMultiple(t *testing.T) {

	test_file, _ := filepath.Abs("../../resources/audio_samples/multiple_keywords.wav")

	p := Porcupine{BuiltInKeywords: []BuiltInKeyword{
		ALEXA, AMERICANO, BLUEBERRY, BUMBLEBEE,
		GRAPEFRUIT, GRASSHOPPER, PICOVOICE, PORCUPINE,
		TERMINATOR}}
	expectedResults := []int{7, 0, 1, 2, 3, 4, 5, 6, 7, 8}

	status, err := p.Init()
	if err != nil {
		t.Fatalf("%v", err)
	}
	if status != SUCCESS {
		t.Fatalf("Porcupine failed to init with PvStatus %d", status)
	}

	t.Logf("Porcupine Version: %s", Version)
	t.Logf("Frame Length: %d", FrameLength)
	t.Logf("Samples Rate: %d", SampleRate)

	data, err := ioutil.ReadFile(test_file)
	if err != nil {
		t.Fatalf("Could not read test file: %v", err)
	}
	data = data[44:] // skip header

	var results []int
	frameLenBytes := FrameLength * 2
	frameCount := int(math.Floor(float64(len(data)) / float64(frameLenBytes)))
	for i := 0; i < frameCount; i++ {
		start := i * frameLenBytes
		count := frameLenBytes

		frame := data[start : start+count]
		result, err := p.Process(frame)
		if err != nil {
			t.Fatalf("Could not read test file: %v", err)
		}
		if result >= 0 {
			t.Logf("Keyword %d triggered at %f", result, float64(i*FrameLength)/float64(SampleRate))
			results = append(results, result)
		}
	}

	for i := range results {
		if results[i] != expectedResults[i] {
			t.Fatalf("Expected keyword %d, but %d was detected.", expectedResults[i], results[i])
		}
	}

	delErr := p.Delete()
	if delErr != nil {
		t.Fatalf("%v", delErr)
	}
}
