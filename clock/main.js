function setClock(){
    var dateInfo = new Date(); 
    var hour = modifyNumber(dateInfo.getHours());
    var min = modifyNumber(dateInfo.getMinutes());
    var sec = modifyNumber(dateInfo.getSeconds());
    var mil = modifyNumber(dateInfo.getMilliseconds())
    if (mil.toString().length !=3){
        mil = "0"+ mil.toString()
    }
    document.getElementById("time").innerHTML = hour + ":" + min  + ":" + sec + ":" + mil;
}
function modifyNumber(time){
    if(parseInt(time)<10){
        return "0"+ time;
    }
    else
        return time;
}
window.onload = function(){
    setClock();
    setInterval(setClock,10); //0.01초마다 setClock 함수 실행
}
