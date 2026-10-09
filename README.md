[guida_git_e_github.md](https://github.com/user-attachments/files/33260221/guida_git_e_github.md)
# 📖 Guida al Workflow di Git e GitHub

Benvenuti in questa repository! Questo file contiene le linee guida essenziali per collaborare al progetto utilizzando Git da terminale. Seguire queste regole ci permetterà di lavorare in modo ordinato, evitare conflitti disastrosi e mantenere il codice sempre funzionante.

---

## 🛑 Le Regole d'Oro del Lavoro in Team

Prima di scrivere qualsiasi comando, è fondamentale comprendere la filosofia di base con cui gestiamo questa repository:

1. **Il branch `main` è intoccabile:** Il `main` (o `master`) contiene *solo* codice testato, funzionante e approvato. **Non si lavora mai e non si fa mai un `push` diretto sul `main`.**
2. **Un branch per ogni funzionalità (e per ogni persona):** Quando devi aggiungere qualcosa di nuovo o risolvere un bug, crea un branch separato. **Non si lavora in più persone sullo stesso branch** contemporaneamente; ognuno deve avere il suo spazio di lavoro isolato.
3. **Nomenclatura chiara dei branch:** Usa nomi descrittivi per i tuoi branch. Esempi: `feature/nuova-homepage`, `bugfix/errore-login`, `docs/aggiornamento-readme`.
4. **Fai commit piccoli e frequenti:** Non aspettare di aver finito un intero sito web per fare un commit. Salva il tuo lavoro passo dopo passo con messaggi chiari (es. *"Aggiunto il bottone di conferma nella pagina contatti"*).
5. **Comunica tramite Pull Request (PR):** Quando hai finito il tuo lavoro sul tuo branch, non fonderlo da solo con il `main`. Apri una Pull Request su GitHub per far revisionare il codice agli altri.

---

## 💻 Guida Passo per Passo ai Comandi da Terminale

Ecco il ciclo di lavoro (workflow) che dovrai seguire ogni volta che inizi a lavorare a una nuova funzionalità.

### 1. Sincronizzare il progetto (Fondamentale!)
Prima di iniziare qualsiasi lavoro, assicurati di avere l'ultima versione del codice presente sul server.
Spostati sul branch principale e scarica gli aggiornamenti:
```bash
git checkout main
git pull origin main
```

### 2. Creare un nuovo branch (Il tuo spazio di lavoro)
Ora che hai il codice aggiornato, crea un nuovo branch isolato dove potrai lavorare senza fare danni. Il comando `-b` crea il branch e ti ci sposta automaticamente dentro:
```bash
git checkout -b nome-del-tuo-branch
```
*(Esempio: `git checkout -b feature/aggiunta-carrello`)*

### 3. Lavorare e controllare lo stato
Lavora sui tuoi file usando il tuo editor di codice preferito. Per essere sicuro di stare lavorando sul branch giusto e per vedere quali file hai modificato, usa **sempre** questo comando:
```bash
git status
```
*Ti mostrerà in rosso i file modificati e ti confermerà in alto su quale branch ti trovi (es. "On branch feature/aggiunta-carrello").*

### 4. Aggiungere le modifiche (Staging)
Una volta che hai completato una parte del lavoro, devi dire a Git quali file modificati vuoi "impacchettare" per il salvataggio.
Per aggiungere **tutti** i file modificati:
```bash
git add .
```
*(Se invece vuoi aggiungere un solo file specifico, usa `git add nome-del-file.ext`)*. Controlla di nuovo con `git status`: i file ora appariranno in verde.

### 5. Salvare le modifiche (Commit)
Ora dai un nome al tuo "pacchetto" di modifiche con un messaggio chiaro:
```bash
git commit -m "Inserisci qui una descrizione chiara di cosa hai fatto"
```

### 6. Inviare le modifiche online (Push)
Ora che il tuo lavoro è salvato localmente sul tuo computer, devi inviarlo a GitHub affinché anche gli altri possano vederlo. 
**Attenzione:** non pushare sul main, ma sul tuo branch!
```bash
git push origin nome-del-tuo-branch
```

---

## 🔀 Cosa succede dopo il Push? (La Pull Request)

Una volta fatto il push, il tuo lavoro non andrà automaticamente nel `main`. 
1. Vai sulla pagina GitHub della repository.
2. Vedrai un pulsante verde con scritto **"Compare & pull request"**.
3. Cliccalo, scrivi una breve descrizione di ciò che hai completato e richiedi l'unione (merge) del tuo branch con il `main`.
4. Un tuo collega (o tu stesso, a seconda delle regole del team) controllerà il codice e, se è tutto ok, approverà il merge.

## 🛠️ Risoluzione dei Problemi Frequenti

* **"Ho sbagliato branch!"**: Se hai modificato dei file ma non hai ancora fatto il commit, puoi cambiare branch senza perdere il lavoro usando `git stash`, spostandoti con `git checkout nome-branch-corretto`, e recuperando i file con `git stash pop`.
* **"Voglio vedere la lista dei branch"**: Usa il comando `git branch`. Il branch con l'asterisco `*` accanto è quello in cui ti trovi attualmente.
