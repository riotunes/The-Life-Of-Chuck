(
s. options. memSize = 1048576;
s. options.numBuffers = 128;
s.options.sampleRate = 96000;  // I tuoi file sono a 96kHz!

s.waitForBoot({
    Buffer.freeAll;
    s.sync;

    ~segmentDur = 30;
    ~fadeTime = 5;
    ~queue = List.new;

    SynthDef(\streamPlayer, { |out=0, buf=0, amp=1, gate=1, fadeTime=5|
        var env = EnvGen.kr(Env.asr(fadeTime, 1, fadeTime), gate, doneAction: 2);
        var rate = BufRateScale.kr(buf);
        var sig = VDiskIn.ar(2, buf, rate, loop: 0);
        Out.ar(out, sig * env * amp);
    }).add;

    s.sync;

    OSCdef(\addFile, { |msg|
        var path = msg[1]. asString;
        if(File.exists(path)) {
            ~queue.add(path);
            ("[QUEUE] Aggiunto: " ++ path. basename).postln;
        } {
            ("[ERROR] File non trovato: " ++ path).postln;
        };
    }, "/addFile");

    Tdef(\djPlayer, {
        // RINOMINATE le chiavi:  syn invece di synth, buffer invece di buf
        var current = (buffer: nil, syn: nil, path: nil);
        var nextPath, nextBuf, nextSynth;

        var freeVoice = { |voice|
            if(voice.syn.notNil) { voice.syn.free };
            if(voice.buffer.notNil) { voice.buffer.free };
        };

        loop {
            // CORRETTO: usa && invece di and:  { }
            if(current.buffer.isNil && ~queue.isEmpty) {
                0.2.wait;
            } {
                if(current.buffer.isNil) {
                    current.path = ~queue.removeAt(0);
                    ("[LOAD] " ++ current.path.basename).postln;
                    current.buffer = Buffer. cueSoundFile(s, current.path, 0, 2, bufferSize: 32768);
                    s.sync;
                    ("[PLAY] >>> " ++ current. path.basename).postln;
                    current.syn = Synth(\streamPlayer, [
                        \buf, current.buffer,
                        \fadeTime, ~fadeTime,
                        \gate, 1
                    ]);
                    (~segmentDur - ~fadeTime).wait;
                } {
                    (~segmentDur - ~fadeTime).wait;
                };

                if(~queue.notEmpty) {
                    nextPath = ~queue.removeAt(0);
                    ("[PRELOAD] " ++ nextPath.basename).postln;
                    nextBuf = Buffer.cueSoundFile(s, nextPath, 0, 2, bufferSize: 32768);
                    s.sync;

                    nextSynth = Synth(\streamPlayer, [
                        \buf, nextBuf,
                        \fadeTime, ~fadeTime,
                        \gate, 1
                    ]);

                    // CORRETTO: usa && invece di and: { }
                    if(current.syn.notNil && current. syn.isPlaying) {
                        current.syn.set(\gate, 0);
                    };

                    ~fadeTime.wait;
                    freeVoice.(current);

                    current = (buffer: nextBuf, syn: nextSynth, path:  nextPath);
                    nextBuf = nil; nextSynth = nil; nextPath = nil;
                } {
                    // CORRETTO: usa && invece di and: { }
                    if(current.syn.notNil && current. syn.isPlaying) {
                        current.syn.set(\gate, 0);
                        ~fadeTime.wait;
                    };
                    freeVoice.(current);
                    current = (buffer:  nil, syn: nil, path: nil);
                };
            };
        }
    }).play;

    "[SYSTEM] READY.  Sample rate: 96kHz.  In attesa su porta 57120... ".postln;
});
)