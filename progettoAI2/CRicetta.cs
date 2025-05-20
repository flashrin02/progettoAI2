using Microsoft.ML.Data;

namespace progettoAI2
{
    public class CRicetta
    {
        private List<string> ner1;

        public int ID { get; set; }
        public string title { get; set; }
        public List<string> ingredients { get; set; }
        public List<string> directions { get; set; }
        public string link { get; set; }
        public string source { get; set; }
        public List<string> ner { get; set; }
        public bool liked { get; set; }

        public CRicetta(int ID, string title, List<string> ingredients, List<string> directions, string link, string source, List<string> ner)
        {
            this.ID = ID;
            this.title = title;
            this.ingredients = ingredients;
            this.directions = directions;
            this.link = link;
            this.source = source;
            this.ner = ner;
            liked = false;
        }

    }

    public class RicettaInput
    {
        public float NumeroIngredienti { get; set; }
        public float NumeroPassaggi { get; set; }

        //Attribuzione che serve a dichiarare la dimensione del vettore (feature vettoriale)
        [VectorType(50)]
        public float[] IngredientiVector { get; set; }

        public bool Label { get; set; }

        public RicettaInput() { }

        public RicettaInput(float numeroIngredienti, float numeroPassaggi, float[] ingredientiVector, bool label)
        {
            NumeroIngredienti = numeroIngredienti;
            NumeroPassaggi = numeroPassaggi;
            IngredientiVector = ingredientiVector;
            Label = label;
        }
    }

    public class RicettaPrediction
    {
        //Per mappare la colonna di output
        [ColumnName("PredictedLabel")]
        public bool predictedLabel;

        public float probability;
        public float score;
    }
}