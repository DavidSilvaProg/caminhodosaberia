<?php
session_start();
if (!isset($_SESSION['id']) || !isset($_SESSION['correcao'])) {
    header("Location: login.html");
    exit();
}
?>

<!DOCTYPE html>
<html lang="pt-BR">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Resultado da Correção - Redação ENEM</title>
    <style>
        /* Mantenha os estilos existentes do corretor.php */
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
        }

        .result-section {
            margin-bottom: 30px;
        }

        .nota-final {
            font-size: 24px;
            font-weight: bold;
            color: #007bff;
            text-align: center;
            margin: 20px 0;
        }

        .competencia {
            margin-bottom: 15px;
            padding: 15px;
            background: #f8f9fa;
            border-radius: 5px;
        }

        .competencia h3 {
            color: #007bff;
            margin-bottom: 10px;
        }

        .competencia .nota {
            font-weight: bold;
            color: #28a745;
        }

        .competencia .comentario {
            font-style: italic;
            color: #6c757d;
        }

        .redacao-original {
            margin-top: 30px;
            padding: 15px;
            background: #e0f7fa;
            border-radius: 5px;
        }

        .redacao-original h3 {
            color: #007bff;
            margin-bottom: 10px;
        }

        .btn-voltar {
            display: inline-block;
            padding: 10px 20px;
            background: #007bff;
            color: white;
            border: none;
            border-radius: 5px;
            font-size: 18px;
            cursor: pointer;
            text-decoration: none;
            margin-top: 20px;
        }

        .btn-voltar:hover {
            background: #0056b3;
        }
    </style>
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

<div class="container2">
    <div class="result-section">
        <h2>Resultado da Correção</h2>
        <div class="nota-final">
            Sua nota: <?php echo $_SESSION['correcao']['nota_final']; ?>/1000
        </div>
        
        <?php foreach ($_SESSION['correcao']['notas'] as $key => $nota): ?>
            <div class="competencia">
                <h3><?php echo $_SESSION['correcao']['criterios'][$key]['nome']; ?></h3>
                <p class="nota">Nota: <?php echo $nota; ?>/200</p>
                <p class="comentario"><?php echo $_SESSION['correcao']['comentarios'][$key]; ?></p>
            </div>
        <?php endforeach; ?>
    </div>
    
    <div class="redacao-original">
        <h3>Sua Redação:</h3>
        <p><strong>Tema:</strong> <?php echo $_SESSION['correcao']['tema']; ?></p>
        <p><?php echo nl2br($_SESSION['correcao']['texto']); ?></p>
    </div>
    
    <a href="corretor.php" class="btn-voltar">Voltar para o Corretor</a>
</div>

<footer>
    <p>&copy; 2024 SCHOLARSUPPORT. Todos os direitos reservados.</p>
    <a href="POLITICA.php">Política de privacidade</a>
</footer>
</body>
</html>