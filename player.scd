(
// ============================================================
// CONFIGURAZIONE SERVER ROBUSTA
// ============================================================

// 1. Aumentiamo drasticamente la memoria (512 MB)
s.options.memSize = 524288;
// 2. Aumentiamo il numero massimo di buffer gestibili
s.options.numBuffers = 2048;

// Esegui il boot. Se è già acceso, assicurati di riavviarlo se cambi memSize!
s.waitForBoot({

    // 3. PULIZIA TOTALE (Fondamentale per evitare memory leak)
    Buffer.freeAll;
    s.sync;

    // ============================================================
    // PARAMETRI
    // ============================================================
    ~segmentDur = 30;
    ~fadeTime   = 2;
    ~queue      = List.new;

    // ============================================================
    // SYNTHDEFS (Mono e Stereo per sicurezza)
    // ============================================================

    // Player Stereo (default)
    SynthDef(\streamPlayer, { |out=0, buf=0, amp=1, gate=1, fadeTime=2|
        var env = EnvGen.kr(Env.asr(fadeTime, 1, fadeTime), gate, doneAction: 2);
        // DiskIn legge da disco, usa poca RAM se il buffer è configurato bene
        var sig = DiskIn.ar(2, buf, loop: 0);
        Out.ar(out, sig * env * amp);
    }).add;

    s.sync;

    // ============================================================
    // OSC HANDLER
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
    // PLAYER LOGIC (Tdef)
    // ============================================================
    Tdef(\djPlayer, {
        var currentBuf, currentSynth;

        loop {
            if (~queue.notEmpty) {
                var path = ~queue.removeAt(0);

                // --- CRUCIALE: TRY-CATCH PER ALLOCAZIONE ---
                try {
                    // Carichiamo il file in modalità streaming.
                    // bufferSize: 32768 è un buon compromesso tra stabilità e RAM.
                    currentBuf = Buffer.cueSoundFile(s, path, 0, 2, bufferSize: 32768);

                    s.sync; // Aspetta che il buffer sia pronto

                    ("[PLAY] >>> " ++ path.basename).postln;

                    currentSynth = Synth(\streamPlayer, [
                        \buf, currentBuf,
                        \fadeTime, ~fadeTime,
                        \gate, 1
                    ]);

                    // Riproduci per la durata del segmento MENO il fade out
                    (~segmentDur - ~fadeTime).wait;

                    // Inizia il fade out
                    if(currentSynth.isPlaying) { currentSynth.set(\gate, 0) };

                    // Aspetta che il suono finisca
                    ~fadeTime.wait;

                } { |error|
                    ("[ERROR] Impossibile riprodurre file: " ++ path).postln;
                    error.postln;
                };

                // --- PULIZIA BUFFER ---
                // Liberiamo il buffer IMMEDIATAMENTE dopo l'uso
                if(currentBuf.notNil) {
                    currentBuf.free;
                    currentBuf = nil;
                };

            } {
                // Se la coda è vuota, polling leggero
                0.2.wait;
            }
        }
    }).play;

    "[SYSTEM] READY. Memoria Allocata: 512MB. In attesa su porta 57120...".postln;
});
)