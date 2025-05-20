using CsvHelper;
using Microsoft.ML;
using Microsoft.ML.Data;
using progettoAI2;
using System.Globalization;
using System.Text.Json;

class Program
{
    static MLContext mlContext = new MLContext(seed: 0);  //Stessi input => stessi risultati
    static ITransformer model;  //Modello ML addestrato
    static PredictionEngine<RicettaInput, RicettaPrediction> predictionEngine;      //Permette di fare previsioni

    static List<CRicetta> ricette = new List<CRicetta>();
    static List<string> vocabolario = new List<string>();

    static List<RicettaInput> trainingSet = new List<RicettaInput>();

    static void Main(string[] args)
    {
        CaricaRicette();
        CostruisciVocabolario();
        CaricaPreferenze();

        //Pipeline serve per preparare dati e addestrare il modello
        var pipeline = mlContext.Transforms.Concatenate("Features", new[] { "NumeroIngredienti", "NumeroPassaggi" })        //Unisce le 2 colonne NumeroIngredienti e NumeroPassaggi in un'unica -> Features
        .Append(mlContext.Transforms.NormalizeMinMax("Features"))           //Normalizzazione delle feature  
        .Append(mlContext.Transforms.Concatenate("FeaturesFinal", new[] { "Features", "IngredientiVector" }))               //Unisce alle feature anche gli ingredienti
        .Append(mlContext.BinaryClassification.Trainers.SdcaLogisticRegression(labelColumnName: "Label", featureColumnName: "FeaturesFinal"));      //Aggiunge l'algoritmo che impara i dati

        //Per aggiungere almeno un esempio fittizio per Fit in caso non ci siano dati
        if (trainingSet.Count == 0)
        {
            trainingSet.Add(CreaDatoFittizio());
        }

        //Per convertire la lista training in un formato leggibile da ML
        var dataView = mlContext.Data.LoadFromEnumerable(trainingSet);
        //Per creare un modello addestrato
        model = pipeline.Fit(dataView);
        //Per fare predizioni
        predictionEngine = mlContext.Model.CreatePredictionEngine<RicettaInput, RicettaPrediction>(model);

        for (int i = 0; i < 10; i++)
        {
            Console.WriteLine("Inserisci gli ingredienti che hai in casa, separati da virgola:");
            string input = Console.ReadLine();
            if (string.IsNullOrWhiteSpace(input))
            {
                Console.WriteLine("Input vuoto, riprova");
                i--;
                continue;
            }

            List<string> ingredientiCasa = new List<string>();
            string[] parts = input.Split(',');
            foreach (string part in parts)
            {
                ingredientiCasa.Add(part.Trim().ToLower());
            }

            List<(CRicetta ricetta, int score)> ricetteConScore = new List<(CRicetta, int)>();
            foreach (var r in ricette)
            {
                int score = ContaIngredientiComuni(r.ingredients, ingredientiCasa);
                if (score > 0)
                {
                    ricetteConScore.Add((r, score));
                }
            }
            //Ordina in modo decrescente di score
            ricetteConScore.Sort((a, b) => b.score.CompareTo(a.score));
            //Prende le prime 10
            List<CRicetta> ricetteTop10 = new List<CRicetta>();
            int max = Math.Min(10, ricetteConScore.Count);
            for (int j = 0; j < max; j++)
            {
                ricetteTop10.Add(ricetteConScore[j].ricetta);
            }

            if (ricetteConScore.Count == 0)
            {
                Console.WriteLine("Nessuna ricetta trovata con quegli ingredienti");
                continue;
            }

            Console.WriteLine("--- Ricette trovate ---");
            int count = 1;
            foreach (var r in ricetteConScore)
            {
                RicettaInput inputRicetta = CreaInputDaRicetta(r.ricetta, vocabolario);

                //Solo se il modello ha dati sufficienti
                bool suggerita = false;
                if (trainingSet.Count >= 2)
                {
                    var pred = predictionEngine.Predict(inputRicetta);
                    suggerita = pred.predictedLabel;
                }

                string messaggio = $"{count}) {r.ricetta.title} (ingredienti in comune: {r.score})";
                if (suggerita)
                {
                    messaggio += " (questo ti potrebbe piacere)";
                }

                Console.WriteLine(messaggio);
                Console.WriteLine($"Ingredienti: {string.Join(", ", r.ricetta.ingredients)}");
                count++;
            }

            Console.WriteLine("Scegli il numero della ricetta che ti piace (o 0 se nessuna):");
            if (!int.TryParse(Console.ReadLine(), out int scelta) || scelta < 0 || scelta > ricetteConScore.Count)
            {
                Console.WriteLine("Input non valido. Salto questa iterazione");
                continue;
            }

            if (scelta == 0)
            {
                Console.WriteLine("Nessuna ricetta scelta in questa iterazione");
                continue;
            }

            CRicetta sceltaRicetta = ricetteConScore[scelta - 1].ricetta;

            Console.Write("Ti piace questa ricetta? (s/n): ");
            string risposta = Console.ReadLine().Trim().ToLower();

            bool label = (risposta == "s" || risposta == "si");
            RicettaInput nuovoDato = CreaInputDaRicetta(sceltaRicetta, vocabolario);
            nuovoDato.Label = label;

            trainingSet.Add(nuovoDato);

            //Riallena modello con tutti i dati aggiornati
            dataView = mlContext.Data.LoadFromEnumerable(trainingSet);
            model = pipeline.Fit(dataView);
            predictionEngine = mlContext.Model.CreatePredictionEngine<RicettaInput, RicettaPrediction>(model);

            Console.WriteLine("Modello aggiornato con il nuovo dato");

            ValutaModello(mlContext, model, trainingSet);


            SalvaPreferenze();
        }
    }

    static RicettaInput CreaDatoFittizio()
    {
        return new RicettaInput(0, 0, new float[50], false);
    }

    static void CaricaRicette()
    {
        string path = "full_dataset_test.csv";
        ricette.Clear();//perchè fai clear ogni volta??? -alessio

        if (File.Exists(path))
        {
            using (var sr = new StreamReader(path))
            using (var csv = new CsvReader(sr, CultureInfo.InvariantCulture))
            {
                csv.Read();
                csv.ReadHeader();

                while (csv.Read())
                {
                    try
                    {
                        int ID = csv.GetField<int>(0);
                        string title = csv.GetField<string>(1).Trim();
                        List<string> ingredients = JsonSerializer.Deserialize<List<string>>(csv.GetField<string>(2));
                        List<string> directions = JsonSerializer.Deserialize<List<string>>(csv.GetField<string>(3));
                        string link = csv.GetField<string>(4);
                        string source = csv.GetField<string>(5);
                        List<string> ner = JsonSerializer.Deserialize<List<string>>(csv.GetField<string>(6));

                        if (string.IsNullOrEmpty(title) || ingredients.Count == 0 || directions.Count == 0 ||
                            string.IsNullOrEmpty(link) || string.IsNullOrEmpty(source) || ner.Count == 0)
                        {
                            Console.WriteLine($"Ricetta con ID {ID} ignorata (dati mancanti)");
                            continue;
                        }

                        ricette.Add(new CRicetta(ID, title, ingredients, directions, link, source, ner));
                    }
                    catch (Exception ex)
                    {
                        Console.WriteLine("Errore lettura ricetta: " + ex.Message);
                    }
                }
            }
        }
        else
        {
            Console.WriteLine("File non trovato");
            return;
        }
    }

    static void CostruisciVocabolario()
    {
        List<string> tuttiIngredienti = new List<string>();

        foreach (var r in ricette)
        {
            foreach (var ing in r.ner)
            {
                string ingrediente = ing.Trim().ToLower();
                if (!tuttiIngredienti.Contains(ingrediente))
                {
                    tuttiIngredienti.Add(ingrediente);
                }
            }
        }

        int IngVocSize = 50;
        //Vocabolario di primi 50 ingredienti
        vocabolario = tuttiIngredienti.Take(IngVocSize).ToList();
    }

    static int ContaIngredientiComuni(List<string> ingr1, List<string> ingr2)
    {
        var set1 = new HashSet<string>();
        foreach (string i in ingr1)
        {
            set1.Add(i.Trim().ToLower());
        }

        var set2 = new HashSet<string>();
        foreach (string i in ingr2)
        {
            set2.Add(i.Trim().ToLower());
        }

        //Lascia nel set1 solo gli elementi in comune a set2
        set1.IntersectWith(set2);
        return set1.Count;
    }


    static RicettaInput CreaInputDaRicetta(CRicetta r, List<string> voc)
    {
        RicettaInput input = new RicettaInput();
        input.NumeroIngredienti = r.ingredients.Count;
        input.NumeroPassaggi = r.directions.Count;

        float[] vettore = new float[voc.Count];
        for (int i = 0; i < voc.Count; i++)
        {
            bool trovato = false;

            foreach (string ing in r.ner)
            {
                if (ing.Trim().ToLower() == voc[i])
                {
                    trovato = true;
                    break;
                }
            }

            if (trovato)
            {
                vettore[i] = 1f;
            }
            else
            {
                vettore[i] = 0f;
            }
        }

        input.IngredientiVector = vettore;
        return input;
    }

    static void CaricaPreferenze()
    {
        trainingSet.Clear();
        if (File.Exists("preferences.json"))
        {
            try
            {
                string json = File.ReadAllText("preferences.json");
                trainingSet = JsonSerializer.Deserialize<List<RicettaInput>>(json);
                Console.WriteLine($"Caricate {trainingSet.Count} preferenze salvate");
            }
            catch
            {
                Console.WriteLine("Errore caricamento preferenze, si parte da 0");
            }
        }
    }

    static void SalvaPreferenze()
    {
        string json = JsonSerializer.Serialize(trainingSet);
        File.WriteAllText("preferences.json", json);
        Console.WriteLine("Preferenze salvate");
    }

    static void ValutaModello(MLContext mlContext, ITransformer model, List<RicettaInput> dati)
    {
        if (dati.Count < 2)
        {
            Console.WriteLine("Non ci sono abbastanza dati per una valutazione");
            return;
        }

        var dataView = mlContext.Data.LoadFromEnumerable(dati);
        var split = mlContext.Data.TrainTestSplit(dataView, testFraction: 0.3); // 30% test set

        // Controlla che ci sia almeno una classe positiva e una negativa nel test set
        var testSet = mlContext.Data.CreateEnumerable<RicettaInput>(split.TestSet, reuseRowObject: false).ToList();
        bool hasPositive = testSet.Any(x => x.Label);
        bool hasNegative = testSet.Any(x => !x.Label);

        if (!hasPositive || !hasNegative)
        {
            Console.WriteLine("Il set di test non contiene sia esempi positivi che negativi. Valutazione saltata.");
            return;
        }

        var predictions = model.Transform(split.TestSet);

        var metrics = mlContext.BinaryClassification.Evaluate(predictions, labelColumnName: "Label");

        Console.WriteLine("=== Metriche del modello ===");
        Console.WriteLine($"Accuracy: {metrics.Accuracy:P2}");
        Console.WriteLine($"Precision: {metrics.PositivePrecision:P2}");
        Console.WriteLine($"Recall: {metrics.PositiveRecall:P2}");
        Console.WriteLine($"F1 Score: {metrics.F1Score:P2}");
        Console.WriteLine("============================");
    }

}
