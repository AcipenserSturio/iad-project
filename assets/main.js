let inference = async (text, result) => {
    const response = await fetch("/infer", {
        method: "POST",
        headers: {
            "Content-type": "application/json"
        },
        body: JSON.stringify({
            text: text.value
        })
    })
    let results_infer = await response.json();
    console.log(results_infer['infer'])
    result.innerHTML = results_infer['infer'];
}

window.onload = function(){
    const text = document.getElementById("text")
    const button = document.getElementById("button")
    const result = document.getElementById("result")
    button.addEventListener("click", ()=>{inference(text, result)})
};