<?php
session_start();

$serverName = 'localhost';
$nomeUsuario = 'root';
$senha = '';
$db = 'db_scholarsupport';

$conn = mysqli_connect($serverName, $nomeUsuario, $senha, $db);

if ($conn == false) {
    echo "erro de conexão";
    exit;
} else {
    $temaRedacao = $_POST['temaRedacao'];
    $redacao = $_POST['redacao'];
    $dataRedacao = date('d/m/Y');
    $idUsuario = $_SESSION["id"];

    // Chamada para a API Python
    $api_url = 'http://localhost:5000/corrigir-redacao'; // Ajuste a URL conforme necessário
    
    $data = [
        'texto' => $redacao,
        'tema' => $temaRedacao
    ];
    
    $options = [
        'http' => [
            'header'  => "Content-type: application/json\r\n",
            'method'  => 'POST',
            'content' => json_encode($data),
        ],
    ];
    
    $context  = stream_context_create($options);
    $result = file_get_contents($api_url, false, $context);
    
    if ($result === FALSE) {
        die('Erro ao chamar a API de correção');
    }
    
    $response = json_decode($result, true);
    
    // Extrair dados da resposta da API
    $notaRedacao = $response['nota_final'];
    $errosRedacao = json_encode($response['comentarios']); // Converte os comentários para JSON para armazenar no banco
    
    // Inserir no banco de dados
    $sql = "INSERT INTO tb_redacao (temaRedacao, redacao, notaRedacao, errosRedacao, dataRedacao, idUsuario) 
            VALUES (?, ?, ?, ?, ?, ?)";
    $stmtInsert = mysqli_prepare($conn, $sql);
    mysqli_stmt_bind_param($stmtInsert, 'ssssss', $temaRedacao, $redacao, $notaRedacao, $errosRedacao, $dataRedacao, $idUsuario);
    
    if (mysqli_stmt_execute($stmtInsert)) {
        // Redirecionar para uma página de resultados com os dados da correção
        $_SESSION['correcao'] = $response;
        header("Location: resultado_redacao.php");
        exit();
    } else {
        echo "Erro ao cadastrar: " . mysqli_error($conn);
    }
}

mysqli_stmt_close($stmtInsert);
mysqli_close($conn);
?>