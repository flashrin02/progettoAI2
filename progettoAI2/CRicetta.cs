using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.ML.Data;

namespace progettoAI2
{
    public class CRicetta
    {
        public int ID { get; set; }
        public string title { get; set; }
        public List<string> ingredients { get; set; }
        public List<string> directions { get; set; }
        public string link { get; set; }
        public string source { get; set; }
        public string ner { get; set; }
        public bool? liked { get; set; }  //Campo opzionale (null se non ancora etichettata)

        public CRicetta(int ID, string nome, List<string> ingredienti, List<string> istruzioni, string link, string source, string ner)
        {
            this.ID = ID;
            title = nome;
            ingredients = ingredienti;
            directions = istruzioni;
            this.link = link;
            this.source = source;
            this.ner = ner;
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
    }

    public class RicettaPrediction
    {
        //Per mappare la colonna di output
        [ColumnName("PredictedLabel")]
        public bool PredictedLabel;

        public float Probability;
        public float Score;
    }
}