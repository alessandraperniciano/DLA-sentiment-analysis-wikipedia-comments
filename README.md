# **DLA-sentiment-analysis-wikipedia-comments**

> Questo è un progetto svolto nell'ambito dell'esame **Deep Learning and Applications Mod.2** del **CdLM in Informatica** presso l'**Università degli Studi di Cagliari**.

| **Studente**          | **Matricola** | **E-Mail**                        |
|-----------------------|---------------|-----------------------------------|
| Alessandra Perniciano | 60/73/xxxxx   | <a.perniciano3@studenti.unica.it> |
| Federico Meloni       | 60/73/xxxxx   | <f.meloni62@studenti.unica.it>    |

Questo progetto universitario si pone l'obiettivo di creare un modello con l'ausilio dei transformers in grado di classificare la tossicità dei commenti su [Wikipedia](https://www.wikipedia.org/).

Per raggiungere l'obiettivo sono stati utilizzati Python e vari framework come PyTorch e BERT.

Il dataset di partenza è stato preso da una [challange](https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge/) pubblicata su [Kaggle](https://www.kaggle.com/) nel 2018 da [Jigsaw](https://jigsaw.google.com/)

<br><br>

---
<br>

## **Dominio ed analisi dei dati**

[Wikipedia](https://www.wikipedia.org/) oltre ad essere "*l'enciclopedia universale online*" integra al suo interno una struttura stile social-network in cui la community del sito può interagire e curare le pagine del sito.

Come ogni altra piattaforma sul web, al suo interno ci sono persone che interagiscono tra di loro in maniera civile ed incivile, il dataset utilizzato mette a disposizione 159.571 e 153.164 commenti della piattaforma rispettivamente per train e test set.

Ogni commento può avere zero o più tra le seguenti label apportate da moderatori umani:
- toxic
- severe_toxic
- obscene
- threat
- insult
- identity_hate

<br>

Come possiamo vedere dagli istogrammi qui sotto siamo davanti ad un dataset altamente sbilanciato (in quanto la maggior parte degli utenti su internet non è tossica):

![Distribuzione](./images/distribution.png)

<br>

Di seguito le parole più comuni nel dataset  

![WordCloud_General](./images/general_wordcloud.png)

Mentre se prendiamo in considerazione solo i record etichettati come tossici  

![WordCloud_Toxic](./images/toxic_wordcloud.png)

<br><br>

---
<br>

## **Preprocessing**

Per poter operare dei dati bisogna prima effettuare delle operazioni di preprocessing.  
Il nostor preprocessing consiste nel manipolare il dataset in modo da eliminare le informazioni inutili o peggio fuorvianti per la rete, oltre all'eliminazione dei dati quel che abbiamo fatto è andare ad aggiungere informazioni (es. POS Tag) e a semplificare le varie frasi (con la lemmatizazione).

### **Pulizia dei dati**

La pulizia dei dati si è divisa in vari sottopassaggi.  
Avendo come dominio il linguaggio naturale dobbiamo tenere conto di tutte le possibili variazioni che possono creare del rumore all'interno del dataset.

Come prima cosa abbiamo portato tutto il dataset in **caratteri minuscoli**, in quanto le lettere maiuscole sono caratteri diversi che veicolano le stesse informazioni di quelli minuscoli.

Dopo di che abbiamo fatto un primo passaggio di standardizzazione del linguaggio **eliminando tutte le contrazioni** presenti in lingua inglese e ponendole in forma estesa (es. "*you're*" -> "*you are*").

Insieme all'eliminazione delle contrazioni abbiamo la **pulizia dello slang** solito dell'internet, andando a sostituire tutte quelle sigle, forme contratte ed abbreviazioni con la loro controparte "canonica" (es. "*m8*" -> "*mate*").

Dopo di che si è passati all'**eliminazione di tutti i caratteri speciali**, in particolare parliamo di simboli, link, tag HTML, caratteri non ASCII.  
Si è deciso quindi di eliminare anche la punteggiatura.

Infine è giunto il momento di **togliere le stopwords**, cioé tutte quelle parole di circostanza che aiutano nella forma ma non veicolano nessuna informazione utile.

Una volta fatto ciò si sono **eliminati quegli ultimi tag rimasti** (es. \r e \n), **i caratteri di spazio ridondanti e quelli ad inizio e fine riga**.

<br>

Questa pulizia è stata eseguita sia sul train set, sia sul test set.  
Qui alcuni esempi prima e dopo la pulizia:
| **Testo originale**                               	| **Testo pulito**                                  	|
|---------------------------------------------------	|---------------------------------------------------	|
| Explanation\r\nWhy the edits made under my use... 	| explanation why edits made username hardcore m... 	|
| COCKSUCKER BEFORE YOU PISS AROUND ON MY WORK      	| cocksucker piss around work                       	|
| I don't anonymously edit articles at all.         	| anonymously edit articles                         	|
| D'aww! He matches this background colour I'm s... 	| daww matches background colour seemingly stuck... 	|
| Please do not add nonsense to Wikipedia. Such ... 	| please add nonsense wikipedia edits considered... 	|

<br>

### **Part of Speech Tagging & Lemmatizazione**
Successivamente alla pulizia del testo si può procedere al Part of Speech Tagging, cioè quell'operazione che associa ad ogni parola un tab tra i seguenti:
- N: noun (nome)
- V: verb (verbo)
- J: adj (aggettivo)
- R: adv (avverbio)

Questo permette all'operazione successiva, la lemmatizazione, di avvenire in maniera migliore.

La lemmatizazione è quell'operazione che porta tutti i sostantivi alla forme base, per esempio i verbi vengono tutti portati all'infinito e gli aggettivi vengono portati tutti alla forma base, andando a modificare eventuali superlativi etc.  
Abbiamo scelto di effettuare la lemmatizazione anziché solo uno stemming in quanto abbiamo valutato che per i nostri scopi informazioni come il tempo verbale non fossero rilevanti, anzi usare più parole per veicolare lo stesso messaggio avrebbe solo aggiunto rumore al nostro dataset.

Alcuni esempi:
| **Testo originale**                               | **Testo pulito**                                  | **POS Tag**                                       | **Lem**                                           |
|---------------------------------------------------|---------------------------------------------------|---------------------------------------------------|---------------------------------------------------|
| Hey man, I'm really not trying to edit war. It... | hey man really trying edit war guy constantly ... | [(hey, n), (man, n), (really, r), (trying, v),... | hey man really try edit war guy constantly rem... |
| Yes, because the mother of the child in the ca... | yes mother child case michael jackson studied ... | [(yes, n), (mother, n), (child, n), (case, n)...  | yes mother child case michael jackson study mo... |
| Before you start throwing accusations and warn... | start throwing accusations warnings let us rev... | [(start, v), (throwing, v), (accusations, n), ... | start throw accusation warning let u review ed... |
| The Mitsurgi point made no sense - why not ar...  | mitsurugi point made sense argue include hindi... | [(mitsurugi, n), (point, n), (made, v), (sense... | mitsurgi point make sense argue include hindi...  |
| Don't mean to bother you \r\n\r\ni see that yo... | mean bother i see writing something regarding ... | [(mean, v), (bother, n), (i, n), (see, v), (wr... | mean bother i see write something regarding re... |

<br>

### **Text Feature Extraction**

Cacca

<br>

### **TD-IDF**

Come ultimo passo del pre-processing viene calcolata la TD-IDF (term frequency-inverse document frequency), una metrica utilizzata per valutare l'importanza di una parola all'interno di un documento rispetto ad un insieme di documenti. Questa è una metrica che viene utilizzata principalmente nell'analisi del testo e nella ricerca dell'informazione.  
La metrica combina due elementi: la frequenza del termine (term frequency, TF) e la frequenza inversa del documento (inverse document frequency, IDF).

La frequenza del termine (TF) rappresenta il numero di volte in cui un termine compare in un documento. Più un termine è presente in un documento, maggiore sarà la sua frequenza.

La frequenza inversa del documento (IDF) tiene in considerazione la rarità di un termine all'interno di un insieme di documenti. Più un termine è raro all'interno di un insieme di documenti, maggiore sarà la sua frequenza inversa.

La metrica TD-IDF combina questi due elementi per valutare l'importanza relativa di un termine all'interno di un documento rispetto ad un insieme di documenti. Un termine con una elevata frequenza all'interno di un documento e una bassa frequenza all'interno di un insieme di documenti avrà un alto valore TD-IDF. Ciò significa che il termine è rilevante per il documento ma poco comune negli altri documenti. Mentre un termine che compare in tutti i documenti avrà un basso valore TD-IDF, in quanto è poco caratterizzante e risulta come una "*costante*" in tutti i documenti che può essere "*semplificata*".

TD-IDF utilizzato spesso per filtrare i termini poco significativi dai documenti, identificare i termini chiave in un documento o gruppo di documenti e anche come feature per algoritmi di apprendimento automatico come classificazione di testo o clustering.

<br>

---
## **Creazione del modello e Fine Tuning**
