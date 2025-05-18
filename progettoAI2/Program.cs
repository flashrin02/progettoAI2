using CsvHelper;
using Microsoft.ML;
using progettoAI2;
using System.Data;
using System.Formats.Asn1;
using System.Globalization;
using System.Runtime.ConstrainedExecution;
using System.Text.Json;

//Caricamento e pulizia dati
string path = "full_dataset.csv";
List<CRicetta> ricette = new List<CRicetta>();
if (File.Exists(path))
{
    using (StreamReader sr = new StreamReader(path))
    using (CsvReader csv = new CsvReader(sr, CultureInfo.InvariantCulture))
    {
        csv.Read();     //Legge la prima riga del file
        csv.ReadHeader();   //Indica che la riga letta è l'intestazione

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
                string ner = csv.GetField<string>(6);

                if (string.IsNullOrEmpty(title) || ingredients.Count == 0 || directions.Count == 0 || string.IsNullOrEmpty(link) || string.IsNullOrEmpty(source) || string.IsNullOrEmpty(ner))
                {
                    Console.WriteLine($"Ricetta con ID {ID} ignorata (titolo mancante o ingredienti vuoti)");
                    continue; //Salta le prossime righe, tornando sul while
                }

                CRicetta ricetta = new CRicetta(ID, title, ingredients, directions, link, source, ner);
                ricette.Add(ricetta);
            }
            catch (Exception ex)
            {
                Console.WriteLine(ex.ToString());
            }
        }
    }
}
else
{
    Console.WriteLine("File non trovato.");
}

//Fase di etichettatura manuale
List<CRicetta> preferite = new List<CRicetta>();
List<CRicetta> nonPiaciute = new List<CRicetta>();

for (int i = 0; i < 50; i++)
{
    CRicetta r = ricette[i];
    Console.WriteLine($"{i + 1}) Titolo: {r.title}");
    Console.WriteLine($"Ingredienti: {string.Join(", ", r.ingredients)}");

    Console.Write("Ti piace questa ricetta? (s/n): ");
    string input = Console.ReadLine().Trim().ToLower();

    if (input == "s" || input == "si")
    {
        r.liked = true;
        preferite.Add(r);
    }
    else
    {
        r.liked = false;
        nonPiaciute.Add(r);
    }
}

//Costruzione vocabolario ingredienti
int IngVocSize = 50;
List<string> tuttiIngredienti = new List<string>();
foreach (CRicetta r in ricette)
{
    foreach (string ing in r.ingredients)
    {
        string clean = ing.Trim().ToLower();
        if (!tuttiIngredienti.Contains(clean))
        {
            tuttiIngredienti.Add(clean);
        }
    }
}

List<string> vocabolario = new List<string>(tuttiIngredienti);
if (vocabolario.Count > IngVocSize)
{
    vocabolario = vocabolario.GetRange(0, IngVocSize);      //Prende i primi 50 ingredienti
}

//Preparazione dati per addestramento
List<RicettaInput> trainingSet = new List<RicettaInput>();
foreach (CRicetta r in preferite)
{
    RicettaInput input = CreaInputDaRicetta(r, vocabolario);
    input.Label = true;
    trainingSet.Add(input);
}

foreach (CRicetta r in nonPiaciute)
{
    RicettaInput input = CreaInputDaRicetta(r, vocabolario);
    input.Label = false;
    trainingSet.Add(input);
}

//Addestramento con ML.NET
MLContext mlContext = new MLContext();
IDataView trainData = mlContext.Data.LoadFromEnumerable(trainingSet);

var pipeline = mlContext.Transforms.Concatenate("Features", new[] { "NumeroIngredienti", "NumeroPassaggi", "IngredientiVector" })
    .Append(mlContext.BinaryClassification.Trainers.FastForest(labelColumnName: "Label", featureColumnName: "Features"));

var model = pipeline.Fit(trainData);

//Predizione
var predictionEngine = mlContext.Model.CreatePredictionEngine<RicettaInput, RicettaPrediction>(model);
List<CRicetta> suggerite = new List<CRicetta>();

for (int i = 50; i < ricette.Count; i++)
{
    CRicetta r = ricette[i];
    RicettaInput input = CreaInputDaRicetta(r, vocabolario);
    RicettaPrediction prediction = predictionEngine.Predict(input);

    if (prediction.PredictedLabel)
    {
        suggerite.Add(r);
    }
}

//Output risultati
Console.WriteLine("\n--- Ricette suggerite ---\n");
int count = 1;
foreach (CRicetta r in suggerite)
{
    Console.WriteLine($"{count}) {r.title}");
    Console.WriteLine($"Ingredienti: {string.Join(", ", r.ingredients)}\n");
    count++;
    if (count > 10) break;
}

static RicettaInput CreaInputDaRicetta(CRicetta r, List<string> vocabolario)
{
    RicettaInput input = new RicettaInput();
    input.NumeroIngredienti = r.ingredients.Count;
    input.NumeroPassaggi = r.directions.Count;

    //Per ogni ingrediente nel vocabolario, controlla se è presente negli ingredienti della ricetta
    float[] vettore = new float[vocabolario.Count];
    for (int i = 0; i < vocabolario.Count; i++)
    {
        string vocabIng = vocabolario[i];
        bool trovato = false;
        foreach (string ing in r.ingredients)
        {
            if (ing.Trim().ToLower() == vocabIng)
            {
                trovato = true;
                break;
            }
        }
        vettore[i] = trovato ? 1f : 0f;
    }
    input.IngredientiVector = vettore;
    return input;
}