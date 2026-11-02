let net;
const webCam = document.getElementById("webcam");
const classifier = knnClassifier.create();
let webcam;

// lista completa con todas las letras de la A a la Z
const lsmClasses = [
    "Letra A", "Letra B", "Letra C", "Letra D", "Letra E", 
    "Letra F", "Letra G", "Letra H", "Letra I", "Letra J", 
    "Letra K", "Letra L", "Letra M", "Letra N", "Letra Ñ", 
    "Letra O", "Letra P", "Letra Q", "Letra R", "Letra S", 
    "Letra T", "Letra U", "Letra V", "Letra W", "Letra X", 
    "Letra Y", "Letra Z"
];

async function app(){
    try{
        document.getElementById("console2").textContent = "Cargando modelo de IA...";
        net = await mobilenet.load();

        webcam = await tf.data.webcam(webCam);
        document.getElementById("console2").textContent = "Modelo listo. Selecciona una letra y registra muestras.";

        while(true){
            const img = await webcam.capture();

            try{
                const result = await net.classify(img);
                const activation = net.infer(img, true);
                
                try{
                    const prediction = await classifier.predictClass(activation);
                    
                    document.getElementById("console2").innerHTML = 
                        '<strong>Seña Detectada:</strong> ' + lsmClasses[prediction.label] + '<br>' +
                        '<strong>Confianza:</strong> ' + (prediction.confidences[prediction.label] * 100).toFixed(2) + "%";
                } catch(error){
                    document.getElementById("console2").textContent = "Agrega ejemplos usando los botones para entrenar el modelo.";
                } finally {
                    activation.dispose();
                }

                document.getElementById("console").innerHTML = 
                    '<strong>Objeto/Entorno:</strong> ' + result[0].className + '<br>' +
                    '<strong>Probabilidad:</strong> ' + (result[0].probability * 100).toFixed(2) + "%";

            } finally {
                img.dispose();
            }
            await tf.nextFrame();
        }
    } catch(error){
        console.error("Error general en la aplicación: ", error);
    }
}

async function addExample(classLetter) {
    if (!net || !webcam) {
        alert('Espera a que el modelo y la cámara estén completamente listos.');
        return;
    }

    // Mapeo dinámico de cualquier letra de la A a la Z con la Ñ incluida
    const alphabet = "ABCDEFGHIJKLMNÑOPQRSTUVWXYZ";
    let classId = alphabet.indexOf(classLetter);

    const img = await webcam.capture();
    try {
        const activation = net.infer(img, true);
        classifier.addExample(activation, classId);
        activation.dispose();
        console.log(`Ejemplo agregado para la seña: ${classLetter}`);
    } finally {
        img.dispose();
    }
}

app();