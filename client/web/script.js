// Driving Data from the Server
var steer_data = [];
var adv_data = [];

// Connecting to ROS
// -----------------

var ros = new ROSLIB.Ros({
url : 'ws://localhost:9090'
});

ros.on('connection', function() {
    console.log('Client has connected to the server!');
    attack(0, $("input[name=flexRadioDefault]:checked").val());
});

ros.on('error', function(error) {
    console.log('Error connecting to websocket server: ', error);
});

ros.on('close', function() {
    console.log('The client has disconnected!');
    $("#customSwitchActivate").prop("checked", false);
});

// Display the original image
var origin_img_listener = new ROSLIB.Topic({
    ros : ros,
    name : '/raw_img',
    messageType : 'std_msgs/String'
  });

origin_img_listener.subscribe(function(msg) {
    $('#origin').attr("src", "data:image/png;base64," + msg.data);
});

// Display the input image
var input_img_listener = new ROSLIB.Topic({
    ros : ros,
    name : '/input_img',
    messageType : 'std_msgs/String'
  });

input_img_listener.subscribe(function(msg) {
    $('#input').attr("src", "data:image/png;base64," + msg.data);
});

// Display the perturbation
var perturb_img_listener = new ROSLIB.Topic({
    ros : ros,
    name : '/perturb_img',
    messageType : 'std_msgs/String'
  });

perturb_img_listener.subscribe(function(msg) {
    $('#diff').attr("src", "data:image/png;base64," + msg.data);
});

// Display the adversarial image
var adv_img_listener = new ROSLIB.Topic({
    ros : ros,
    name : '/adv_img',
    messageType : 'std_msgs/String'
  });

adv_img_listener.subscribe(function(msg) {
    $('#adv').attr("src", "data:image/png;base64," + msg.data);
});

// Receive Training result for UAPr
// socket.on('unir_train', function (data) {
//     $("#train_res").text("Train: " + parseFloat(data.absolute).toFixed(2) + ' ' + parseFloat(data.percentage).toFixed(2) + "%");
// });

// Activate the attack
var attack_pub = new ROSLIB.Topic({
    ros : ros,
    name : '/attack',
    messageType : 'std_msgs/Int32'
});

function attack(isAttack, type) {
    if(isAttack)
    {
        var attack_type = 0;
        if(type === 'fgsmr_left')
        {
            attack_type = 1;
        }
        if(type === 'fgsmr_right')
        {
            attack_type = 2;
        }
        var attack_msg = new ROSLIB.Message({
            data: attack_type
        });
        attack_pub.publish(attack_msg);
    }
    else
    {
        var attack_msg = new ROSLIB.Message({
            data: 0
        });
        attack_pub.publish(attack_msg);
    }
}

// Chart Options
var options = {
    series: [
        {
            name: "Without attack",
            data: steer_data.slice()
        },
        {
            name: "With attack",
            data: adv_data.slice()
        },
    ],
    chart: {
        id: 'realtime',
        height: 350,
        type: 'line',
        animations: {
            enabled: true,
            easing: 'linear',
            dynamicAnimation: {
                speed: 1000
            }
        },
        toolbar: {
            show: false
        },
        zoom: {
            enabled: false
        }
    },
    colors: ['#77B6EA', '#545454'],
    dataLabels: {
        enabled: false
    },
    stroke: {
        curve: 'smooth'
    },
    title: {
        text: 'Steering Angle',
        align: 'left',
        style: {
            fontSize: '20px'
        }
    },
    markers: {
        size: 0
    },
    xaxis: {
        type: 'line',
        labels: {
            show: false
        }
        // range: XAXISRANGE,
    },
    yaxis: {
        min: -200,
        max: 200,
        labels: {
            style: {
                fontSize: '18px',
            }
        },
        decimalsInFloat: 1,
        tickAmount: 10
    },
    legend: {
        fontSize: '22px',
        position: 'top',
        horizontalAlign: 'right',
        floating: true,
        offsetY: -25,
        offsetX: -5
    }
};

// Attack Deactivated
function resume() {
    $('#diff').attr("src", "./hold.png");
    $('#adv').attr("src", "./hold.png");
}

$(document).ready(function () {

    // Select different attacks
    $("input[name=flexRadioDefault]").change(function () {
        attack(0, this.value);
        $("#customSwitchActivate").prop("checked", false);
        $("#origin").css("border-style", "none");
    });

    // Activate different attacks
    $('#customSwitchActivate').change(function () {
        if ($(this).prop('checked')) {
            attack(1, $("input[name=flexRadioDefault]:checked").val());
            $("#origin").css("border-style", "solid");
            $("#origin").css("border-color", "coral");
            $("#customSwitchTrain").prop("checked", false);
        }
        else {
            attack(0, $("input[name=flexRadioDefault]:checked").val());
            $("#origin").css("border-style", "none");
            resume();
        }
    })

    // Activate traning / learning for Universal Adversarial Perturbation
    // $('#customSwitchTrain').change(function () {
    //     if ($(this).prop('checked')) {
    //         attack(1, $("input[name=flexRadioDefault]:checked").val() + '_train');
    //         $("#origin").css("border-style", "solid");
    //         $("#origin").css("border-color", "coral");
    //         $("#customSwitchActivate").prop("checked", false);
    //     }
    //     else {
    //         attack(0, $("input[name=flexRadioDefault]:checked").val() + '_train');
    //         $("#origin").css("border-style", "none")
    //     }
    // })

    var chart = new ApexCharts(document.querySelector("#chart"), options);
    chart.render();

    var listener = new ROSLIB.Topic({
        ros : ros,
        name : '/cmd_vel_attack',
        messageType : 'std_msgs/Float64MultiArray'
    });
    
    listener.subscribe(function(message) {
        window.data = message.data
        console.log('Received message on ' + listener.name + ': ' + message.data[0]);
        // $("#attack_res").text("Attack: From " + parseFloat(message.angular.z).toFixed(2) + ' to ' + parseFloat(message.linear.z).toFixed(2) );

        steer_data.push(message.data[0] * 100);
        adv_data.push(message.data[1] * 100);
        if (steer_data.length > 50) {
            steer_data.shift();
        }
        if (adv_data.length > 50) {
            adv_data.shift();
        }

        chart.updateSeries([{ data: steer_data }, { data: adv_data }]);
        // listener.unsubscribe();
    });
});
