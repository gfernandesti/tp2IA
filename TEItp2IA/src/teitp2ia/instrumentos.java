/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package teitp2ia;

import java.io.FileReader;
import java.util.Random;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.trees.Id3;
import weka.core.Instance;
import weka.core.Instances;

/**
 *
 * @author gabriel
 */
public class instrumentos {
     public static void main(String[] args) throws Exception {
    // Lendo os exemplos a partir do arquivo instrumentos.arff
		FileReader leitor = new FileReader("instrumentos.arff");
		Instances instrumentos = new Instances(leitor);
		
		// Definindo o índice do atributo classe (último atributo do conjunto)
		instrumentos.setClassIndex(instrumentos.numAttributes() - 1);
		
		// Criando uma nova base com os exemplos embaralhados 
		instrumentos = instrumentos.resample(new Random());			
		
		// Abordagem Hold out de validação cruzada 
		Instances baseTeste = instrumentos.testCV(2, 0); // Obtendo subconjunto para testes
		Instances baseTreino = instrumentos.trainCV(2, 0); // Obtendo subconjunto para treinamento
		
		// Criando os classificadores que serão avaliados
		       Id3 arvore = new Id3(); // 
                       NaiveBayes naive = new NaiveBayes(); //
		
		// Treinando os classificadores instanciados
		arvore.buildClassifier(baseTreino);
		naive.buildClassifier(baseTreino);
		
		System.out.println("real\tarvore\tnaive"); // imprimindo rótulos para as colunas
		
		for (int e = 0; e < baseTeste.numInstances(); e++) {
			Instance exemplo = baseTeste.instance(e);
			System.out.print(exemplo.classValue()); // imprimindo o valor da classe real do exemplo
			exemplo.setClassMissing(); // removendo informação da classe
			double classe = arvore.classifyInstance(exemplo); // resposta da arvore
			System.out.print("\t" + classe); // imprimindo resposta da arvore
			classe = naive.classifyInstance(exemplo); // resposta do naive
			System.out.println("\t" + classe); // imprimindo resposta do naive
		}
     }
    
}
