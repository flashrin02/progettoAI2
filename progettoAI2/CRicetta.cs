using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace progettoAI2
{
    internal class CRicetta
    {
        public int ID { get; set; }
        public string title { get; set; }
        public List<string> Ingredients { get; set; }
        public List<string> directions { get; set; }
        public string link { get; set; }

        public string source { get; set; }

        public string ner { get; set; }
        public CRicetta(int ID, string nome, List<string> ingredienti, List<string> istruzioni, string link, string source, string ner)
        {
            this.ID = ID;
            title = nome;
            Ingredients = ingredienti;
            directions = istruzioni;
            this.link = link;
            this.source = source;
            this.ner = ner;
        }
        public void StampaRicetta()
        {
            Console.WriteLine($"Nome: {title}");
            Console.WriteLine("Ingredienti:");
            foreach (var ingrediente in Ingredients)
            {
                Console.WriteLine($"- {ingrediente}");
            }
            Console.WriteLine("Istruzioni:");
            foreach (var istruzione in directions)
            {
                Console.WriteLine($"- {istruzione}");
            }
            Console.WriteLine($"Link: {link}");
            Console.WriteLine($"Source: {source}");
        }
    }
}
/////////////////