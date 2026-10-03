Procedure comuni Git (workflow)
Per mantenere la repository pulita e funzionante, è fondamentale seguire questo flusso di lavoro per ogni modifica.

1. Sincronizzazione iniziale (pull)
È vitale sincronizzarsi con la repo remota OGNI SINGOLA VOLTA prima di iniziare a lavorare. Questo assicura di lavorare sulla versione più recente del progetto.

Esegui: git pull origin main
2. Creazione e nomenclatura del branch
Lavora sempre separatamente rispetto al ramo principale creando un tuo branch dedicato.

Esegui: git checkout -b <categoria>/<nome>/<breve descrizione>
Regole di nomenclatura del branch:

<categoria>: Utilizza il nome della cartella su cui stai lavorando scegliendo tra code, electrical, inventor o kicad.
<nome>: Il tuo nome scritto tutto in lettere minuscole (es. luca).
<breve descrizione>: Una breve descrizione in 2-5 parole delle modifiche che andrai a eseguire (es. added-tokenizer-for-input-data).
3. Salvataggio locale (add e commit)
Effettua i commit per ogni set di modifiche significative.

Aggiungi i file modificati alla staging area: git add . (per tutti i file) oppure git add <nome_file> (per file singoli).
Controlla lo stato dei file: git status.
Salva le modifiche: git commit -m "Breve descrizione di cosa hai modificato".
4. Sincronizzazione pre-push e push
Prima di caricare le modifiche sul server, assicurati che nessuno abbia modificato il main nel frattempo per evitare potenziali conflitti.

Esegui di nuovo: git pull origin main
Manda le modifiche online sul tuo branch: git push origin <categoria>/<nome>/<breve descrizione>
5. Pull request (PR) e merging
Una volta terminato il lavoro sul tuo branch, apri una Pull Request su GitHub.

Base branch: main
Compare branch: Il tuo branch di lavoro
Crea una Classic Pull request se è pronto per la revisione.
La revisione e l'approvazione finale (con eventuale merge o richiesta di modifiche) verranno effettuate da Kiko Miletos.
