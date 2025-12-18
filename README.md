# CPAC-Hackaton
# 🎬 The Life of Chuck: A Speculative Mirror

> *"I contain multitudes."* — Walt Whitman (Ispirato al film 'The Life of Chuck')

**The Life of Chuck** è un'installazione digitale interattiva sviluppata per il **CPAC Hackathon 2025**. Il progetto invita l'utente a un momento di profonda introspezione, creando un ponte tra il presente e i propri sogni futuri attraverso un'esperienza visiva e narrativa personalizzata.

---

## 🌟 Visione del Progetto
Ispirato alla struttura narrativa di Stephen King, dove ogni individuo è un universo intero, questa demo punta a:
* **Riflessione Attiva:** Visualizzare come le intenzioni di oggi plasmano il panorama del domani.
* **Identità Centrale:** L'utente è il fulcro fisico della scena, mentre il mondo intorno a lui muta in base ai suoi sogni.
* **Introspezione Generativa:** Utilizzare input testuali per seminare una narrazione visiva del possibile.

## 🛠️ Caratteristiche Tecniche
L'esperienza è stata ottimizzata per essere fluida e immersiva nonostante la scala ridotta:

* **Cinematic Background Engine:** Gestione di background animati tramite cycling di frame ad alta risoluzione, ottimizzati per non gravare sulla CPU durante l'uso della webcam.
* **Interfaccia Minimalista:** Design "All-White on Black" con tipografia *Georgia* per richiamare l'estetica dei romanzi e dei titoli di testa cinematografici.
* **Data Persistence:** Sistema di salvataggio immediato (`user_data.txt`) per catturare ogni riflessione dell'utente in tempo reale.
* **Mirror Flip Logic:** Correzione della distorsione della webcam per un'esperienza "a specchio" naturale.

## 🚀 Installazione e Uso

### Prerequisiti
* Python 3.11+
* Webcam funzionante

### Setup
1. Clona il repository:
   ```bash
   git clone [https://github.com/DavidGadiaga/CPAC-Hackaton.git](https://github.com/DavidGadiaga/CPAC-Hackaton.git)
   cd CPAC-Hackaton
