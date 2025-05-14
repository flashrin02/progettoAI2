using progettoAI2;

string path = "full_dataset.csv";
List<CRicetta> ricette = new List<CRicetta>();
if (File.Exists(path))
{
    using (StreamReader sr = new StreamReader(path))
    {
        string line;
        bool isHeader = true;       //Per saltare la prima riga di intestazione
        while ((line = sr.ReadLine()) != null)
        {
            if (isHeader)
            {
                isHeader = false;
                continue;
            }

            string[] parts = line.Split(',');

            int ID = int.Parse(parts[0]);
            string title = parts[1];
            List<string> ingredients = parts[2].Split(',').ToList();
            List<string> directions = parts[2].Split(',').ToList();
            string link = parts[4];
            string source = parts[5];
            string ner = parts[6];
            CRicetta ricetta = new CRicetta(ID, title, ingredients, directions, link, source, ner);
            ricette.Add(ricetta);
        }
    }
}
else
{
    Console.WriteLine("File non trovato.");
}