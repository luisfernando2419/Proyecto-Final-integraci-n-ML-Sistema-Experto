namespace MachineLearning

// El código implementa un sistema experto para recomendar carreras universitarias, combinando reglas fijas y Machine Learning.
// Se basa en las habilidades del usuario en matemáticas, ciencias, arte y comunicación, ingresadas manualmente.
// Primero, utiliza reglas explícitas para sugerir carreras predefinidas (por ejemplo, altos puntajes en matemáticas y ciencias sugieren ingeniería).
// Si no se cumplen las reglas, un modelo de Machine Learning entrenado con **ML.NET** predice la carrera más adecuada basándose en patrones aprendidos de datos históricos.
// Además, cualquier caso no clasificado se almacena para mejorar el modelo en futuros reentrenamientos, permitiendo que el sistema evolucione dinámicamente.
{
    using System;
    using System.Collections.Generic;
    using Microsoft.ML;
    using Microsoft.ML.Data;

    namespace SistemaExpertoCarreras
    {
        // Datos de entrada
        public class CareerData
        {
            [LoadColumn(0)] public float Matematicas { get; set; }
            [LoadColumn(1)] public float Ciencias { get; set; }
            [LoadColumn(2)] public float Arte { get; set; }
            [LoadColumn(3)] public float Comunicacion { get; set; }
            [LoadColumn(4)] public string Carrera { get; set; }
        }

        // Resultado del modelo
        public class CareerPrediction
        {
            [ColumnName("PredictedLabel")]
            public string PredictedCarrera { get; set; }
        }

        class Program
        {
            static void Main(string[] args)
            {
                // Contexto de ML.NET
                MLContext mlContext = new MLContext();

                // Conjunto de datos inicial
                var trainingData = new List<CareerData>
            {
                new CareerData { Matematicas = 9, Ciencias = 8, Arte = 3, Comunicacion = 5, Carrera = "Ingeniería" },
                new CareerData { Matematicas = 5, Ciencias = 7, Arte = 4, Comunicacion = 8, Carrera = "Medicina" },
                new CareerData { Matematicas = 4, Ciencias = 3, Arte = 8, Comunicacion = 7, Carrera = "Diseño Gráfico" },
                new CareerData { Matematicas = 6, Ciencias = 5, Arte = 7, Comunicacion = 8, Carrera = "Publicidad" },
                new CareerData { Matematicas = 9, Ciencias = 8, Arte = 2, Comunicacion = 4, Carrera = "Economía" },
            };

                // Cargar los datos
                IDataView dataView = mlContext.Data.LoadFromEnumerable(trainingData);

                // Construcción del pipeline
                var pipeline = mlContext.Transforms.Conversion.MapValueToKey("Label", "Carrera")
                    .Append(mlContext.Transforms.Concatenate("Features", "Matematicas", "Ciencias", "Arte", "Comunicacion"))
                    .Append(mlContext.MulticlassClassification.Trainers.SdcaMaximumEntropy("Label", "Features"))
                    .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

                // Entrenar el modelo
                var model = pipeline.Fit(dataView);

                // Crear el predictor
                var predictor = mlContext.Model.CreatePredictionEngine<CareerData, CareerPrediction>(model);

                // Sistema experto: reglas fijas combinadas con ML
                Console.WriteLine("Ingrese sus puntajes (1-10) en las siguientes áreas:");
                Console.Write("Matemáticas: ");
                float matematicas = float.Parse(Console.ReadLine());
                Console.Write("Ciencias: ");
                float ciencias = float.Parse(Console.ReadLine());
                Console.Write("Arte: ");
                float arte = float.Parse(Console.ReadLine());
                Console.Write("Comunicación: ");
                float comunicacion = float.Parse(Console.ReadLine());

                // Crear el caso nuevo
                var nuevoCaso = new CareerData
                {
                    Matematicas = matematicas,
                    Ciencias = ciencias,
                    Arte = arte,
                    Comunicacion = comunicacion
                };

                // Reglas predefinidas del sistema experto
                if (matematicas > 8 && ciencias > 7)
                {
                    Console.WriteLine("Regla: Ingeniería es una carrera recomendada.");
                }
                else if (arte > 7 && comunicacion > 6)
                {
                    Console.WriteLine("Regla: Diseño Gráfico o Publicidad son opciones recomendadas.");
                }
                else
                {
                    // Usar ML para recomendaciones adicionales
                    var prediccion = predictor.Predict(nuevoCaso);
                    Console.WriteLine($"ML: Carrera sugerida: {prediccion.PredictedCarrera}");
                }

                // Almacenar datos nuevos para reentrenamiento
                trainingData.Add(new CareerData
                {
                    Matematicas = matematicas,
                    Ciencias = ciencias,
                    Arte = arte,
                    Comunicacion = comunicacion,
                    Carrera = "Sin clasificar"
                });

                Console.WriteLine("Datos no clasificados almacenados para mejorar el modelo.");
            }
        }
    }


}
