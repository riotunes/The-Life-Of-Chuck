(
// ============================================================
// CONFIGURAZIONE SERVER ROBUSTA
// ============================================================

s.options.memSize   = 1048576; // 1 GB
s.options.numBuffers = 2048;

s.waitForBoot({

    // Pulizia iniziale
    Buffer.freeAll;
    s.sync;

    // ============================================================
    // PARAMETRI
    // ============================================================
    ~segmentDur = 30;     // durata totale per traccia
    ~fadeTime   = 5;      // fade-in e fade-out
    ~queue      = List.new;

    // ============================================================
    // SYNTHDEF (stereo)
    // ============================================================
    SynthDef(\streamPlayer, { |out=0, buf=0, amp=1, gate=1, fadeTime=5|
        var env = EnvGen.kr(Env.asr(fadeTime, 1, fadeTime), gate, doneAction: 2);
        var sig = DiskIn.ar(2, buf, loop: 0);
        Out.ar(out, sig * env * amp);
    }).add;

    s.sync;

    // ============================================================
    // OSC HANDLER: /addFile <path>
    // ============================================================
    OSCdef(\addFile, { |msg|
        var path = msg[1].asString;
        if(File.exists(path)) {
            ~queue.add(path);
            ("[QUEUE] Aggiunto: " ++ path.basename).postln;
        } {
            ("[ERROR] File non trovato: " ++ path).postln;
        };
    }, "/addFile");

    // ============================================================
    // PLAYER LOGIC (Tdef con crossfade)
    // ============================================================
    Tdef(\djPlayer, {
        var current = (buf: nil, synth: nil, path: nil);
        var nextPath, nextBuf, nextSynth;

        // Helper per liberare buffer e synth
        var freeVoice = { |voice|
            if(voice.synth.notNil) { voice.synth.free };
            if(voice.buf.notNil)   { voice.buf.free };
        };

        loop {
            // Se non stiamo suonando nulla, attendi un file in coda
            if(current.buf.isNil and: { ~queue.isEmpty }) {
                0.2.wait;
                continue;
            };

            // Se non c'è nulla in riproduzione, prendi il primo disponibile
            if(current.buf.isNil) {
                current.path = ~queue.removeAt(0);
                ("[LOAD] " ++ current.path.basename).postln;
                current.buf = Buffer.cueSoundFile(s, current.path, 0, 2, bufferSize: 32768);
                s.sync;
                ("[PLAY] >>> " ++ current.path.basename).postln;
                current.synth = Synth(\streamPlayer, [
                    \buf, current.buf,
                    \fadeTime, ~fadeTime,
                    \gate, 1
                ]);
                // Attendi il punto di overlap (segmentDur - fadeTime)
                (~segmentDur - ~fadeTime).wait;
            } {
                // Stiamo suonando: aspettiamo il punto di overlap
                (~segmentDur - ~fadeTime).wait;
            };

            // Prova a caricare il prossimo se c'è
            if(~queue.notEmpty) {
                nextPath = ~queue.removeAt(0);
                ("[PRELOAD] " ++ nextPath.basename).postln;
                nextBuf = Buffer.cueSoundFile(s, nextPath, 0, 2, bufferSize: 32768);
                s.sync;

                // Avvia il prossimo con fade-in mentre il corrente va in fade-out
                nextSynth = Synth(\streamPlayer, [
                    \buf, nextBuf,
                    \fadeTime, ~fadeTime,
                    \gate, 1
                ]);

                // Inizia fade-out del corrente
                if(current.synth.notNil and: { current.synth.isPlaying }) {
                    current.synth.set(\gate, 0);
                };

                // Attendi il fade-out
                ~fadeTime.wait;

                // Libera il corrente
                freeVoice.(current);

                // Promuovi il next a corrente
                current = (buf: nextBuf, synth: nextSynth, path: nextPath);
                nextBuf = nil; nextSynth = nil; nextPath = nil;

                // Continua il loop: partirà la nuova finestra di 25s prima del prossimo overlap
            } {
                // Nessun prossimo: chiudi il corrente e libera, poi torna a idle
                if(current.synth.notNil and: { current.synth.isPlaying }) {
                    current.synth.set(\gate, 0);
                    ~fadeTime.wait;
                };
                freeVoice.(current);
                current = (buf: nil, synth: nil, path: nil);
                // Torna allo stato di attesa
            };
        }
    }).play;

    "[SYSTEM] READY. Memoria Allocata: 512MB. In attesa su porta 57120...".postln;
});
)