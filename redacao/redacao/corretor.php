<?php  
// Verifica se está logado
session_start();
if (!isset($_SESSION['id'])) {
    header("Location: login.html");
    exit();
}
$id = $_SESSION["id"];
?>

<!DOCTYPE html>
<html lang="pt-BR">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Redação ENEM</title>
    <style>
        body, h1, ul, li, p {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: Arial, sans-serif;
            line-height: 1.6;
        }

        header {
            width: 100%;
            height: 80px;
            background: rgba(255, 182, 193, 0.5);
            padding: 20px;
        }

        .container {
            display: flex;
            justify-content: center;
            align-items: center;
        }

        .logo img {
            height: 70px;
            margin-right: 15px;
        }

        .title {
            text-align: center;
            font-size: 35px;
            color: black;
            font-family: 'Times New Roman', Times, serif;
        }

        nav {
            background-color: #007bff;
            padding: 10px 0;
        }

        nav ul {
            list-style: none;
            display: flex;
            justify-content: center;
        }

        nav ul li {
            margin-right: 20px;
        }

        nav ul li a {
            color: white;
            text-decoration: none !important;
            font-size: 16px;
            padding: 5px 10px;
            display: block;
        }

        nav ul li a:hover {
            background-color: #0056b3;
            border-radius: 5px;
        }

        footer {
            background: rgba(255, 182, 193, 0.5);
            color: #fff;
            text-align: center;
            padding: 10px 0;
            position: relative !important;
            width: 100%;
            bottom: 0;
            margin-left: -9px;
        }

        footer p {
            font-size: 14px;
            color: #333;
        }

        footer a {
            color: #007bff;
            text-decoration: none;
            font-size: 14px;
        }

        footer a:hover {
            text-decoration: underline;
        }

        .container2 {
            width: 80%;
            margin: auto;
            padding: 20px;
            background: white;
            box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
            margin-top: 20px;
            display: flex;
        }
        .left-section {
            flex: 2;
            padding-right: 20px;
        }
        .right-section {
            flex: 1;
            background: #e0f7fa;
            padding: 20px;
            border-radius: 5px;
            margin-left: 25px;
        }
        textarea, input[type="text"] {
            width: 100%;
            padding: 10px;
            margin-bottom: 10px;
            border: 1px solid #ccc;
            border-radius: 5px;
            font-size: 16px;
        }
        button {
            display: block;
            width: 100%;
            padding: 10px;
            background: #007bff;
            color: white;
            border: none;
            border-radius: 5px;
            font-size: 18px;
            cursor: pointer;
        }
        button:hover {
            background: #0056b3;
        }
        #contador {
            font-size: 14px;
            color: #555;
            text-align: right;
        }

       .obs{
        color: red;
        text-align: justify;
       }

    </style>
    <script>
        function contarCaracteres() {
            var texto = document.getElementById("redacao").value;
            document.getElementById("contador").innerText = "Caracteres: " + texto.length;
        }
    </script>
</head>
<body>
<header>
    <div class="container">
        <div class="logo">
            <img src="logonova.png" alt="Logo">
        </div>
        <h1 class="title">CAMINHO DO SABER</h1>
    </div>
</header>
<nav>
    <ul>
        <li><a href="home.php">Home</a></li>
        <li><a href="exibirProvas.php">Provas</a></li>
        <li><a href="corretor.php">Corretor</a></li>
        <li><a href="progresso.php">Progresso</a></li>
        <li><a href="perfil.php">Perfil</a></li>
    </ul>
</nav>
<form action="redacao.php" method="POST">
    <div class="container2">
        <div class="left-section">
            <h2>Insira seu TEMA e sua REDAÇÃO:</h2>
            <input type="text" name="temaRedacao" id="temaRedacao" placeholder="Digite o tema da redação..." required>
            <textarea rows="15" name="redacao" id="redacao" placeholder="Digite sua redação aqui..." oninput="contarCaracteres()"></textarea>
            <p id="contador">Caracteres: 0</p>
            <button type="submit">Enviar Redação</button>
        </div>
        <div class="right-section">
            <h3>Critérios para nota 1000 no ENEM (cada critério vale 200 pontos):</h3>
            <ul>
                <li><strong>1. Domínio da norma culta:</strong> Gramática e ortografia corretas.</li>
                <li><strong>2. Compreensão do tema:</strong> Abordar exatamente o que é pedido.</li>
                <li><strong>3. Coesão textual:</strong> Uso adequado de conectivos e estrutura bem organizada.</li>
                <li><strong>4. Argumentação:</strong> Apresentar ideias consistentes e bem desenvolvidas.</li>
                <li><strong>5. Proposta de intervenção:</strong> Deve ser clara, viável e detalhada.</li>
            </ul>
            <br>
            <p class="obs"><strong>Observação:</strong> A redação ideal possui quatro parágrafos ( um de introdução, dois de desenvolvimento e um de conclusão) com média de 1800 a 2000 caracteres.</p>
        </div>
    </div>
</form>
<footer>
    <p>&copy; 2024 SCHOLARSUPPORT. Todos os direitos reservados.</p>
    <a href="POLITICA.php">Política de privacidade</a>
</footer>
</body>
</html>